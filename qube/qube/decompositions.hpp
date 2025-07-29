#pragma once
/*!
@file
Decomposistion functions.
 */

#include "debug.hpp"

#include <qpp/qpp.hpp>
#include "maths/arithmetic.hpp"
#include "maths/gtest_macros.hpp"

#include <unsupported/Eigen/EulerAngles>
#include <unsupported/Eigen/MatrixFunctions>

#include <numbers>
#include <ranges>

namespace qube
{

    //! @brief Compute rotation parameters from unitary matrix
    auto unitary_to_rotation(qube::maths::Matrix auto const& U)
    {
        using namespace std::literals::complex_literals;
        using namespace std::numbers;

        /* U = exp(iH), with H hermitian */
        auto const H = Eigen::Matrix2cd{ -1.i * U.log() };
        EXPECT_MATRIX_CLOSE(H, H.adjoint(), 1e-12);

        auto const alpha_c = 0.5 * H.trace();
        EXPECT_LT(std::abs(alpha_c.imag()), 1e-12);
        auto const alpha = std::remainder(alpha_c.real(), 2. * pi);

        auto const H_2 = (H - Eigen::Vector2cd::Constant(alpha_c).asDiagonal().toDenseMatrix()).eval();
        EXPECT_MATRIX_CLOSE(H_2, H_2.adjoint(), 1e-12);
        EXPECT_LT(std::abs(H_2.trace()), 1e-12);

        auto const H_22 = (H_2 * H_2).eval();
        EXPECT_TRUE(H_22.isDiagonal(1e-12));
        EXPECT_COMPLEX_CLOSE(H_22(0,0), H_22(1,1), 1e-12);
        EXPECT_COMPLEX_CLOSE(H_22(0,0), H_22(0,0).real(), 1e-12);
        EXPECT_GT(H_22(0,0).real(), -1e-12);

        auto const theta = 2. * std::sqrt(std::abs(H_22(0,0).real()));

        auto const n_dot_sigma = (H_2 / (-0.5 * theta)).eval();
        EXPECT_MATRIX_CLOSE((n_dot_sigma * n_dot_sigma), Eigen::Matrix2cd::Identity(), 1e-12);

        auto const n = Eigen::Vector3d
        {
            n_dot_sigma(1, 0).real(),
            n_dot_sigma(1, 0).imag(),
            n_dot_sigma(0, 0).real(),
        };
        EXPECT_COMPLEX_CLOSE(n.squaredNorm(), 1., 1.e-12);
        EXPECT_MATRIX_CLOSE(n_dot_sigma, (n[0] * qpp::gt.X + n[1] * qpp::gt.Y + n[2] * qpp::gt.Z).eval(), 1e-12);

        auto const rotation = (std::exp(1.i * alpha) * qpp::gt.Rn(theta, { n[0], n[1], n[2] })).eval();
        EXPECT_MATRIX_CLOSE(rotation, U, 1e-12);

        debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
        debug() << ">> H:\n" << qpp::disp(H) << "\n\n";
        debug() << ">> alpha: " << alpha << "\n\n";
        debug() << ">> H_2:\n" << qpp::disp(H_2) << "\n\n";
        debug() << ">> H_22:\n" << qpp::disp(H_22) << "\n\n";
        debug() << ">> theta: " << theta << "\n\n";
        debug() << ">> n_dot_sigma:\n" << qpp::disp(n_dot_sigma) << "\n\n";
        debug() << ">> n: " << qpp::disp(n.transpose()) << "\n\n";
        debug() << ">> rotation:\n" << qpp::disp(rotation) << "\n\n";

        return std::tuple{ alpha, theta, n };
    }

    //! @brief Compute Euler angles for a compatible input M (rotation matrix, unit quaternion, ...)
    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis>
    auto euler_angles(auto const& M)
    {
        using scalar_t = std::decay_t<decltype(M)>::Scalar;
        using system_t = Eigen::EulerSystem<_AlphaAxis, _BetaAxis, _GammaAxis>;
        using angles_t = Eigen::EulerAngles<scalar_t, system_t>;

        return angles_t{ M };
    }

    //! @brief Compute Euler decomposition and phase from phase, angle and unit axis
    //! @see unitary_to_rotation (output)
    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis, bool reentering = false>
    auto euler_decomposition(qube::maths::RealNumber auto const& alpha, qube::maths::RealNumber auto const& theta, qube::maths::Matrix auto const& n)
    {
        using namespace std::numbers;

        auto const q = Eigen::Quaterniond{ Eigen::AngleAxisd{theta, n} };
        auto const euler = euler_angles<_AlphaAxis, _BetaAxis, _GammaAxis>(q);

        if constexpr (!reentering)
        {
            if (!Eigen::Quaterniond{ euler }.isApprox(q, 1e-12))
            {
                auto const new_alpha = std::remainder(alpha + pi, 2. * pi);
                return euler_decomposition<_AlphaAxis, _BetaAxis, _GammaAxis, true>(new_alpha, theta + 2.*pi, n);
            }
        }

        auto res = Eigen::Vector4d{};
        res[0] = alpha;
        res.tail<3>() = euler.angles();

        return res;
    }

    //! @brief Compute Euler generalized decomposition of a rotation matrix R
    //! @details The unit axis of rotation are not necessarily orthogonal
    //! Such a decomposition does not always exist (see necessary and sufficient condition in the code)
    //! Return NaN if the condition is not met
    //! Computed from "On Coordinate-Free Rotation Decomposition: Euler Angles about Arbitrary Axes" by Giulia Piovan and Francesco Bullo
    auto generalized_euler_decomposition(
        qube::maths::Matrix auto const& R
        , qube::maths::Matrix auto const& r1
        , qube::maths::Matrix auto const& r2
        , qube::maths::Matrix auto const& r3)
    {
        auto theta = Eigen::Vector3d::Zero().eval();

        auto const r2xr3 = r2.cross(r3).eval();
        auto const r2xr2xr3 = r2.cross(r2xr3).eval();

        auto const a = -r1.dot(r2xr2xr3);
        auto const b = r1.dot(r2xr3);
        auto const c = r1.dot(R * r3 - r3 - r2xr2xr3);

        theta[1] = std::atan2(b, a) + std::atan2(std::sqrt(a * a + b * b - c * c), c);

        auto const collinear = qube::maths::collinear((R.transpose() * r1).eval(), r3, 1e-12, true);

        if(collinear)
        {
            auto const w2 = (R.transpose() * r2).eval();
            theta[2] = -std::atan2(w2.dot(r3.cross(r2)), r2.dot(w2) - r2.dot(r3) * w2.dot(r3));
        }
        else
        {
            auto const r2xr1 = r2.cross(r1).eval();
            auto const r2xr2xr1 = r2.cross(r2xr1).eval();

            auto const cos = std::cos(theta[1]);
            auto const sin = std::sin(theta[1]);
            auto const v1 = (r3 + sin * r2xr3 + (1. - cos) * r2xr2xr3).eval();
            auto const w1 = (R * r3).eval();
            auto const v3 = (r1 - sin * r2xr1 + (1. - cos) * r2xr2xr1).eval();
            auto const w3 = (R.transpose() * r1).eval();

            theta[0] =  std::atan2(w1.dot(r1.cross(v1)), v1.dot(w1) - v1.dot(r1) * w1.dot(r1));
            theta[2] = -std::atan2(w3.dot(r3.cross(v3)), v3.dot(w3) - v3.dot(r3) * w3.dot(r3));
        }

        auto const condition_left = std::abs(r1.dot((R - r2 * r2.transpose()) * r3));
        auto const condition_right = std::sqrt((1. - std::pow(r1.dot(r2), 2)) * (1. - std::pow(r3.dot(r2), 2)));

        EXPECT_EQ((condition_left > condition_right), theta.hasNaN());

        debug() << ">> condition_left : " << condition_left << "\n";
        debug() << ">> condition_right : " << condition_right << "\n";
        debug() << ">> condition met : " << std::boolalpha << (condition_left <= condition_right) << "\n";
        debug() << ">> theta : " << qpp::disp(theta.transpose()) << "\n";

        return theta;
    }

    //! @brief Compute Euler generalized decomposition from phase, angle and unit axis
    //! @see generalized_euler_decomposition
    template < bool reentering = false >
    auto generalized_euler_decomposition(
        qube::maths::RealNumber auto const& alpha, qube::maths::RealNumber auto const& theta, qube::maths::Matrix auto const& n
        , qube::maths::Matrix auto const& r1
        , qube::maths::Matrix auto const& r2
        , qube::maths::Matrix auto const& r3)
    {
        using namespace std::numbers;

        auto const q = Eigen::Quaterniond{ Eigen::AngleAxisd{theta, n} };
        auto const euler = generalized_euler_decomposition(q.toRotationMatrix(), r1, r2, r3);

        if constexpr (!reentering)
        {
            auto const computed_q = Eigen::AngleAxisd(euler[0], r1) * Eigen::AngleAxisd(euler[1], r2) * Eigen::AngleAxisd(euler[2], r3);
            if (!computed_q.isApprox(q, 1e-12))
            {
                auto const new_alpha = std::remainder(alpha + pi, 2. * pi);
                return generalized_euler_decomposition<true>(new_alpha, theta + 2.*pi, n, r1, r2, r3);
            }
        }

        auto res = Eigen::Vector4d{};
        res[0] = alpha;
        res.tail<3>() = euler;

        return res;
    }

    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis>
    auto abc_decomposition(qube::maths::Matrix auto const& U)
    {
        using namespace std::literals::complex_literals;

        static_assert(
            _AlphaAxis == Eigen::EULER_Z
            && _BetaAxis == Eigen::EULER_Y
            && _GammaAxis == Eigen::EULER_Z,
            "abc_decomposition is only implemented for ZYZ euler anlges");

        auto const [alpha, theta, n] = unitary_to_rotation(U);

        auto const e = euler_decomposition<_AlphaAxis, _BetaAxis, _GammaAxis>(alpha, theta, n);

        auto const rotation = (std::exp(1.i * e[0]) * qpp::gt.RZ(e[1]) * qpp::gt.RY(e[2]) * qpp::gt.RZ(e[3])).eval();
        EXPECT_MATRIX_CLOSE(rotation, U, 1e-12);

        auto const A = (qpp::gt.RZ(e[1]) * qpp::gt.RY(0.5 * e[2])).eval();
        auto const B = (qpp::gt.RY(-0.5 * e[2]) * qpp::gt.RZ(-0.5 * (e[3]+e[1]))).eval();
        auto const C = qpp::gt.RZ(0.5 * (e[3]-e[1]));
        auto const ABC = (A * B * C).eval();
        EXPECT_MATRIX_CLOSE(ABC, Eigen::Matrix2cd::Identity(), 1e-12);

        auto const B_inverse_signs = (qpp::gt.RY(0.5 * e[2]) * qpp::gt.RZ(0.5 * (e[3]+e[1]))).eval();
        auto const XBX = (qpp::gt.X * B * qpp::gt.X).eval();
        EXPECT_MATRIX_CLOSE(XBX, B_inverse_signs, 1e-12);

        auto const AXBXC = (A * qpp::gt.X * B * qpp::gt.X * C).eval();
        auto const eiaAXBXC = (std::exp(1.i * e[0]) * AXBXC).eval();
        EXPECT_MATRIX_CLOSE(eiaAXBXC, U, 1e-12);

        debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
        debug() << ">> alpha: " << alpha << "\n\n";
        debug() << ">> euler: " << qpp::disp(e.transpose()) << "\n\n";
        debug() << ">> A:\n" << qpp::disp(A) << "\n\n";
        debug() << ">> B:\n" << qpp::disp(B) << "\n\n";
        debug() << ">> C:\n" << qpp::disp(C) << "\n\n";
        debug() << ">> ABC:\n" << qpp::disp(ABC) << "\n\n";
        debug() << ">> AXBXC:\n" << qpp::disp(AXBXC) << "\n\n";
        debug() << ">> eiaAXBXC:\n" << qpp::disp(eiaAXBXC) << "\n\n";

        return std::tuple{ e[0], A, B, C };
    }

    template < typename Scalar = Eigen::dcomplex >
    struct two_level_matrix_t
    {
        Eigen::Index index = 0;
        Eigen::Matrix2<Scalar> U = Eigen::Matrix2<Scalar>::Identity();
    };

    template < typename Scalar >
    Eigen::MatrixX<Scalar> operator*(qube::maths::Matrix auto const& M, two_level_matrix_t<Scalar> const& s)
    {
        auto const n = M.cols();
        auto P = Eigen::MatrixX<Scalar>::Identity(n, n).eval();
        auto const& i = s.index;
        P({i, i+1}, {i, i+1}) = s.U;

        debug() << ">> P:\n" << qpp::disp(P) << "\n\n";

        return M * P;
    }

    template < typename Scalar >
    auto two_level_unitary_decomposition(maths::Matrix auto const& U, std::vector<two_level_matrix_t<Scalar>>& out, Eigen::Index const offset = 0) -> void
    {
        using namespace Eigen::indexing;

        debug() << ">> U:\n" << qpp::disp(U) << "\n\n";

        auto const n = U.cols();
        assert(n >= 2);
        assert(n == U.rows());

        if(n == 2)
        {
            auto const I = seqN(fix<0>, fix<2>);
            out.emplace_back(offset, U(I,I));
            return;
        }

        auto const range = std::views::iota(0, n-1) | std::views::common;

        auto const I2 = Eigen::Matrix2<Scalar>::Identity();
        auto const I = Eigen::MatrixX<Scalar>::Identity(n, n);
        EXPECT_MATRIX_CLOSE((U.adjoint()*U).eval(), I, 1.e-12);
        EXPECT_MATRIX_CLOSE((U*U.adjoint()).eval(), I, 1.e-12);

        auto new_U = U.eval();
        for(auto&& i : range | std::views::reverse)
        {
            auto const& b = new_U(i+1, 0);
            if (b == 0.)
                continue;
            auto const& a = new_U(i, 0);

            auto s = two_level_matrix_t<Scalar>{};
            s.index = i + offset;
            s.U.col(0) = Eigen::Vector2<Scalar>{ a, b };
            s.U.col(1) = Eigen::Vector2<Scalar>{ -std::conj(b), std::conj(a) };
            s.U /= std::sqrt(std::norm(a) + std::norm(b));
            EXPECT_MATRIX_CLOSE((s.U.adjoint()*s.U).eval(), I2, 1.e-12);
            EXPECT_MATRIX_CLOSE((s.U*s.U.adjoint()).eval(), I2, 1.e-12);

            out.emplace_back(s);
            new_U({i, i+1}, all) = s.U.adjoint() * new_U({i, i+1}, all);
        }
        debug() << ">> new_U:\n" << qpp::disp(new_U) << "\n\n";

        return two_level_unitary_decomposition(new_U(lastN(n-1), lastN(n-1)), out, offset + 1);
    }

    template < typename Scalar = Eigen::dcomplex >
    auto two_level_unitary_decomposition(maths::Matrix auto const& U)
    {
        auto result = std::vector<two_level_matrix_t<Scalar>>{};
        two_level_unitary_decomposition(U, result);
        return result;
    }

} /* namespace qube */
