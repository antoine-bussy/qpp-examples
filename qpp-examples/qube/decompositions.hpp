#pragma once
/*!
@file
Decomposistion functions.
 */

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>

#include <unsupported/Eigen/EulerAngles>
#include <unsupported/Eigen/MatrixFunctions>

#include <numbers>

namespace qpp_e::qube
{

    //! @brief Compute rotation parameters from unitary matrix
    template < bool print_text = false >
    auto unitary_to_rotation(qpp_e::maths::Matrix auto const& U)
    {
        using namespace std::literals::complex_literals;
        using namespace std::numbers;

        /* U = exp(iH), with H hermitian */
        auto const H = Eigen::Matrix2cd{ -1.i * U.log() };
        EXPECT_MATRIX_CLOSE(H, H.adjoint(), 1e-12);

        auto const alpha_c = 0.5 * H.diagonal().sum();
        if(std::abs(alpha_c.real()) > 1e-12)
        {
            EXPECT_COMPLEX_CLOSE(alpha_c, alpha_c.real(), 1e-12);
        }
        auto const alpha = std::remainder(alpha_c.real(), 2. * pi);

        auto const H_2 = (H - Eigen::Vector2cd::Constant(alpha_c).asDiagonal().toDenseMatrix()).eval();
        EXPECT_MATRIX_CLOSE(H_2, H_2.adjoint(), 1e-12);
        EXPECT_COMPLEX_CLOSE(H_2(1,1), -H_2(0,0), 1e-12);

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

        if constexpr (print_text)
        {
            std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
            std::cerr << ">> H:\n" << qpp::disp(H) << "\n\n";
            std::cerr << ">> alpha: " << alpha << "\n\n";
            std::cerr << ">> H_2:\n" << qpp::disp(H_2) << "\n\n";
            std::cerr << ">> H_22:\n" << qpp::disp(H_22) << "\n\n";
            std::cerr << ">> theta: " << theta << "\n\n";
            std::cerr << ">> n_dot_sigma:\n" << qpp::disp(n_dot_sigma) << "\n\n";
            std::cerr << ">> n: " << qpp::disp(n.transpose()) << "\n\n";
            std::cerr << ">> rotation:\n" << qpp::disp(rotation) << "\n\n";
        }

        return std::tuple{ alpha, theta, n };
    }

    //! @brief Compute Euler angles for a compatible input M (rotation matrix, unit quaternion, ...)
    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis, bool print_text = false>
    auto euler_angles(auto const& M)
    {
        using scalar_t = std::decay_t<decltype(M)>::Scalar;
        using system_t = Eigen::EulerSystem<_AlphaAxis, _BetaAxis, _GammaAxis>;
        using angles_t = Eigen::EulerAngles<scalar_t, system_t>;

        return angles_t{ M };
    }

    //! @brief Compute Euler decomposition and phase from phase, angle and unit axis
    //! @see unitary_to_rotation (output)
    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis, bool print_text = false>
    auto euler_decomposition(qpp_e::maths::RealNumber auto const& alpha, qpp_e::maths::RealNumber auto const& theta, qpp_e::maths::Matrix auto const& n)
    {
        using namespace std::numbers;

        auto const q = Eigen::Quaterniond{ Eigen::AngleAxisd{theta, n} };
        auto const euler = euler_angles<_AlphaAxis, _BetaAxis, _GammaAxis>(q);
        auto const qq = Eigen::Quaterniond{ euler };

        auto res = Eigen::Vector4d{};
        /* Eigen Euler Angles module performs range conversion, which messes up with the "single-qubit Euler decomposition" */
        res[0] = qq.isApprox(q, 1e-12) ? alpha : (alpha + pi);
        res.tail<3>() = euler.angles();

        return res;
    }

    //! @brief Compute Euler generalized decomposition of a rotation matrix R
    //! @details The unit axis of rotation are not necessarily orthogonal
    //! Such a decomposition does not always exist (see necessary and sufficient condition in the code)
    //! Return NaN if the condition is not met
    //! Computed from "On Coordinate-Free Rotation Decomposition: Euler Angles about Arbitrary Axes" by Giulia Piovan and Francesco Bullo
    template < bool print_text = false >
    auto generalized_euler_decomposition(
        qpp_e::maths::Matrix auto const& R
        , qpp_e::maths::Matrix auto const& r1
        , qpp_e::maths::Matrix auto const& r2
        , qpp_e::maths::Matrix auto const& r3)
    {
        auto theta = Eigen::Vector3d::Zero().eval();

        auto const r2xr3 = r2.cross(r3).eval();
        auto const r2xr2xr3 = r2.cross(r2xr3).eval();

        auto const a = -r1.dot(r2xr2xr3);
        auto const b = r1.dot(r2xr3);
        auto const c = r1.dot(R * r3 - r3 - r2xr2xr3);

        theta[1] = std::atan2(b, a) + std::atan2(std::sqrt(a * a + b * b - c * c), c);

        auto const collinear = qpp_e::maths::collinear((R.transpose() * r1).eval(), r3, 1e-12, true);

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

        if constexpr (print_text)
        {
            std::cerr << ">> condition_left : " << condition_left << "\n";
            std::cerr << ">> condition_right : " << condition_right << "\n";
            std::cerr << ">> condition met : " << std::boolalpha << (condition_left <= condition_right) << "\n";
            std::cerr << ">> theta : " << qpp::disp(theta.transpose()) << "\n";
        }

        return theta;
    }

    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis, bool print_text = false>
    auto abc_decomposition(qpp_e::maths::Matrix auto const& U)
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

        if constexpr (print_text)
        {
            std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
            std::cerr << ">> alpha: " << alpha << "\n\n";
            std::cerr << ">> euler: " << qpp::disp(e.transpose()) << "\n\n";
            std::cerr << ">> A:\n" << qpp::disp(A) << "\n\n";
            std::cerr << ">> B:\n" << qpp::disp(B) << "\n\n";
            std::cerr << ">> C:\n" << qpp::disp(C) << "\n\n";
            std::cerr << ">> ABC:\n" << qpp::disp(ABC) << "\n\n";
            std::cerr << ">> AXBXC:\n" << qpp::disp(AXBXC) << "\n\n";
            std::cerr << ">> eiaAXBXC:\n" << qpp::disp(eiaAXBXC) << "\n\n";
        }

        return std::tuple{ e[0], A, B, C };
    }

} /* namespace qpp_e::qube */
