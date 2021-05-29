#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

#include <numbers>
#include <execution>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Test of Q++ mket function
TEST(chapter1_4, mket)
{
    using namespace qpp::literals;

    EXPECT_MATRIX_EQ(qpp::mket({0, 0, 0}), 000_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 0, 1}), 001_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 1, 0}), 010_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 1, 1}), 011_ket);

    EXPECT_MATRIX_EQ(qpp::mket({1, 0, 0}), 100_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 0, 1}), 101_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 1, 0}), 110_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 1, 1}), 111_ket);
}

//! @brief Figure 1.14
TEST(chapter1_4, toffoli_gate)
{
    using namespace qpp::literals;

    EXPECT_MATRIX_EQ(qpp::gt.TOF * 000_ket, 000_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 001_ket, 001_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 010_ket, 010_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 011_ket, 011_ket);

    EXPECT_MATRIX_EQ(qpp::gt.TOF * 100_ket, 100_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 101_ket, 101_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 110_ket, 111_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 111_ket, 110_ket);

    for(auto&& a : { 0u, 1u })
        for(auto&& b : { 0u, 1u })
            for(auto&& c : { 0u, 1u })
                EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::mket({a, b, c}), qpp::mket({a, b, (c + a * b) % 2}));

    auto toffoli_matrix = Eigen::Matrix<qpp::cplx, 8, 8>::Identity().eval();
    toffoli_matrix(Eigen::lastN(2), Eigen::lastN(2)) = Eigen::Vector4cd{ 0, 1, 1, 0 }.reshaped(2,2);
    EXPECT_MATRIX_EQ(qpp::gt.TOF, toffoli_matrix);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::gt.TOF.adjoint(), (Eigen::Matrix<qpp::cplx, 8, 8>::Identity()));

    if constexpr (print_text)
        std::cerr << ">> Toffoli gate:\n" << qpp::disp(qpp::gt.TOF) << '\n';
}

//! @brief Figure 1.15
TEST(chapter1_4, nand)
{
    for(auto&& a : { 0u, 1u })
        for(auto&& b : { 0u, 1u })
            EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::mket({a, b, 1}), qpp::mket({a, b, !(a && b)}));
}

//! @brief Figure 1.16
TEST(chapter1_4, fanout)
{
    for(auto&& a : { 0u, 1u })
        EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::mket({1, a, 0}), qpp::mket({1, a, a}));
}

namespace
{
    auto consteval functions()
    {
        return std::array
        {
            std::array{ 0u, 0u },
            std::array{ 0u, 1u },
            std::array{ 1u, 0u },
            std::array{ 1u, 1u },
        };
    }

    auto matrix(auto const& f)
    {
        auto F = Eigen::Matrix2cd::Zero().eval();
        F(f[0], 0) = F(f[1], 1) = 1;
        return F;
    }

    auto matrixU(auto const& F)
    {
        using namespace qpp::literals;

        auto Uf = Eigen::Matrix4cd::Zero().eval();
        Uf.col(0) = qpp::kron(0_ket, F * 0_ket);
        Uf.col(1) = qpp::kron(0_ket, qpp::gt.X * F * 0_ket);
        Uf.col(2) = qpp::kron(1_ket, F * 1_ket);
        Uf.col(3) = qpp::kron(1_ket, qpp::gt.X * F * 1_ket);
        return Uf;
    }
}

//! @brief Figure 1.17
TEST(chapter1_4, function)
{
    using namespace qpp::literals;

    for(auto&& f : functions())
    {
        auto const F = matrix(f);
        EXPECT_MATRIX_EQ(F * 0_ket, qpp::mket({f[0]}));
        EXPECT_MATRIX_EQ(F * 1_ket, qpp::mket({f[1]}));

        auto const Uf = matrixU(F);
        EXPECT_MATRIX_EQ(Uf * Uf.adjoint(), Eigen::Matrix4cd::Identity());
        EXPECT_MATRIX_EQ(Uf, Uf.adjoint());

        EXPECT_MATRIX_EQ(Uf * 00_ket, qpp::mket({0u, f[0]}));
        EXPECT_MATRIX_EQ(Uf * 01_ket, qpp::mket({0u, (1u + f[0]) % 2u}));
        EXPECT_MATRIX_EQ(Uf * 10_ket, qpp::mket({1u, f[1]}));
        EXPECT_MATRIX_EQ(Uf * 11_ket, qpp::mket({1u, (1u + f[1]) % 2u}));

        if constexpr (print_text)
        {
            std::cerr << "-----------------------------\n";
            std::cerr << ">> F:\n" << qpp::disp(F) << '\n';
            std::cerr << ">> Uf:\n" << qpp::disp(Uf) << '\n';
        }
    }
}

//! @brief Figure 1.17 and equation 1.37
TEST(chapter1_4, function_parallelism)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    for(auto&& f : functions())
    {
        auto const Uf = matrixU(matrix(f));

        auto const x = ((0_ket + 1_ket) * inv_sqrt2).eval();
        auto const y = 0_ket;

        auto const psi = (Uf * qpp::kron(x, y)).eval();
        auto const expected_psi = ((qpp::mket({0u, f[0]}) + qpp::mket({1u, f[1]})) * inv_sqrt2).eval();
        EXPECT_MATRIX_EQ(psi, expected_psi);

        if constexpr (print_text)
        {
            std::cerr << "-----------------------------\n";
            std::cerr << ">> Uf:\n" << qpp::disp(Uf) << '\n';
        }
    }
}

//! @brief Equation 1.38 and figure 1.18
TEST(chapter1_4, hadamard_transform_2d)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto const x = ((0_ket + 1_ket) * inv_sqrt2).eval();
    auto const hadamard_transform = (0.5 * (00_ket + 01_ket + 10_ket + 11_ket)).eval();
    EXPECT_MATRIX_CLOSE(hadamard_transform, qpp::kron(x, x), 1e-12);

    auto const H2 = qpp::kron(qpp::gt.H, qpp::gt.H);
    EXPECT_MATRIX_CLOSE(hadamard_transform, H2 * 00_ket, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Hadamard transform:\n" << qpp::disp(hadamard_transform) << '\n';
        std::cerr << ">> H2 matrix:\n" << qpp::disp(H2) << '\n';
    }
}

//! @brief Equation 1.39
TEST(chapter1_4, hadamard_transform)
{
    for(auto&& n : { 3, 4, 5 })
    {
        auto const _2_pow_n = std::pow(2, n);
        auto const inv_sqrt_2_pow_n = 1. / std::sqrt(_2_pow_n);
        auto const hadamard_transform = (inv_sqrt_2_pow_n * Eigen::VectorXcd::Ones(_2_pow_n)).eval();
        auto const Hn = qpp::kronpow(qpp::gt.H, n);
        EXPECT_MATRIX_CLOSE(hadamard_transform, Hn * qpp::mket(std::vector<qpp::idx>(n, 0u)), 1e-12);

        if constexpr (print_text)
        {
            std::cerr << "-----------------------------\n";
            std::cerr << ">> Number of Qubits:" << n << '\n';
            std::cerr << ">> Hadamard transform:\n" << qpp::disp(hadamard_transform) << '\n';
            std::cerr << ">> Hn matrix:\n" << qpp::disp(Hn) << '\n';
        }
    }
}

namespace
{
    auto random_function(int n)
    {
        auto gen = std::default_random_engine{};
        auto d = std::uniform_int_distribution<qpp::idx>{ 0u, 1u };
        auto f = std::vector<qpp::idx>(std::pow(2, n));
        for (auto& v : f)
            v = d(gen);
        return f;
    }

    auto matrix_n(auto const& f)
    {
        auto const _2_pow_n = f.size();
        auto F = Eigen::MatrixXcd::Zero(2, _2_pow_n).eval();
        for (auto&& i : std::views::iota(0u, _2_pow_n))
            F(f[i], i) = 1;
        return F;
    }

    auto matrixU_n(auto const& F)
    {
        auto const _2_pow_n = F.cols();
        auto const non_F = (qpp::gt.X * F).eval();

        auto Uf = Eigen::MatrixXcd::Zero(_2_pow_n * 2, _2_pow_n * 2).eval();
        for (auto&& i : std::views::iota(0, _2_pow_n))
        {
            auto const x = Eigen::VectorXcd::Unit(_2_pow_n, i);
            Uf.col(2*i) = qpp::kron(x, F * x);
            Uf.col(2*i+1) = qpp::kron(x, non_F * x);
        }
        return Uf;
    }
}

//! @brief Equation 1.40
TEST(chapter1_4, function_7d)
{
    using namespace qpp::literals;

    auto constexpr n = 7;
    auto constexpr _2_pow_n = static_cast<qpp::idx>(std::pow(2, n));
    auto constexpr policy = std::execution::par;

    auto const f = random_function(n);

    auto const F = matrix_n(f);
    for (auto&& i : std::views::iota(0u, _2_pow_n))
        EXPECT_MATRIX_EQ(F * Eigen::VectorXcd::Unit(_2_pow_n, i), qpp::mket({f[i]}));

    auto const Uf = matrixU_n(F);
    EXPECT_MATRIX_EQ(Uf * Uf.adjoint(), Eigen::MatrixXcd::Identity(2 * _2_pow_n, 2 * _2_pow_n));
    EXPECT_MATRIX_EQ(Uf, Uf.adjoint());

    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    static_assert(std::ranges::random_access_range<decltype(range)>);

    std::for_each(policy, range.begin(), range.end(),
        [&](auto&& i)
    {
        auto const x = Eigen::VectorXcd::Unit(_2_pow_n, i);
        EXPECT_MATRIX_EQ(Uf * qpp::kron(x, 0_ket), qpp::kron(x, qpp::mket({f[i]})));
        EXPECT_MATRIX_EQ(Uf * qpp::kron(x, 1_ket), qpp::kron(x, qpp::mket({1u - f[i]})));
    });

    auto const hadamard_state = qpp::kron(qpp::kronpow(qpp::gt.H, n) * qpp::mket(std::vector<qpp::idx>(n, 0u)), 0_ket);
    auto const out_state = (Uf * hadamard_state).eval();
    auto const expected_out_state = (std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::VectorXcd::Zero(2 * _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        auto const x = Eigen::VectorXcd::Unit(_2_pow_n, i);
        return qpp::kron(x, qpp::mket({f[i]}));
    }) / std::sqrt(_2_pow_n)).eval();
    EXPECT_MATRIX_CLOSE(out_state, expected_out_state, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << "-----------------------------\n";
        std::cerr << ">> F:\n" << qpp::disp(F) << '\n';
        std::cerr << ">> Uf:\n" << qpp::disp(Uf) << '\n';
        std::cerr << ">> Hadamard state:\n" << qpp::disp(hadamard_state) << '\n';
        std::cerr << ">> Result state:\n" << qpp::disp(out_state) << '\n';
    }
}

//! @brief Figure 1.19 and equations 1.41 through 1.45
TEST(chapter1_4, deutsch_algorithm)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    for(auto&& f : functions())
    {
        auto const Uf = matrixU(matrix(f));

        auto const psi0 = 01_ket;

        auto const psi1 = (qpp::kronpow(qpp::gt.H, 2) * psi0).eval();
        EXPECT_MATRIX_CLOSE(psi1, 0.5 * qpp::kron(0_ket + 1_ket, 0_ket - 1_ket), 1e-12);

        auto const psi2 = (Uf * psi1).eval();
        if(f[0] == f[1])
            EXPECT_MATRIX_CLOSE(psi2, std::pow(-1., f[0]) * 0.5 * qpp::kron(0_ket + 1_ket, 0_ket - 1_ket), 1e-12);
        else
            EXPECT_MATRIX_CLOSE(psi2, std::pow(-1., f[0]) * 0.5 * qpp::kron(0_ket - 1_ket, 0_ket - 1_ket), 1e-12);

        auto const psi3 = qpp::apply(psi2, qpp::gt.H, { 0 });
        if(f[0] == f[1])
            EXPECT_MATRIX_CLOSE(psi3, std::pow(-1., f[0]) * inv_sqrt2 * qpp::kron(0_ket, 0_ket - 1_ket), 1e-12);
        else
            EXPECT_MATRIX_CLOSE(psi3, std::pow(-1., f[0]) * inv_sqrt2 * qpp::kron(1_ket, 0_ket - 1_ket), 1e-12);

        EXPECT_MATRIX_CLOSE(psi3, std::pow(-1., f[0]) * inv_sqrt2 * qpp::kron(qpp::mket({ (f[0] + f[1]) % 2 }), 0_ket - 1_ket), 1e-12);

        if constexpr (print_text)
        {
            std::cerr << "-----------------------------\n";
            std::cerr << ">> f(0) = " << f[0] << ", f(1) = " << f[1] << '\n';
            std::cerr << ">> Uf:\n" << qpp::disp(Uf) << '\n';
            std::cerr << ">> psi0:\n" << qpp::disp(psi0) << '\n';
            std::cerr << ">> psi1:\n" << qpp::disp(psi1) << '\n';
            std::cerr << ">> psi2:\n" << qpp::disp(psi2) << '\n';
            std::cerr << ">> psi3:\n" << qpp::disp(psi3) << '\n';
        }
    }
}

namespace
{
    auto random_constant_function(int n)
    {
        auto const f = random_function(0);
        return std::vector<qpp::idx>(std::pow(2, n), f[0]);
    }
    auto random_balanced_function(int n)
    {
        auto f = std::vector<qpp::idx>(std::pow(2, n), 0u);
        std::fill(f.begin() + f.size() / 2, f.end(), 1u);
        std::shuffle(f.begin(), f.end(), std::default_random_engine{});
        return f;
    }
}

//! @brief Test of bitwise inner product with std::popcount
TEST(chapter1_4, bitwise_inner_product)
{
    auto constexpr n = 7u;
    auto constexpr _2_pow_n = static_cast<qpp::idx>(std::pow(2, n));
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto const dims = std::vector<qpp::idx>(n, 2u);

    for (auto&& x : range)
    {
        auto const x_vec = qpp::n2multiidx(x, dims);
        auto const X = Eigen::VectorX<qpp::idx>::Map(x_vec.data(), x_vec.size());

        for(auto&& z: range)
        {
            auto const z_vec = qpp::n2multiidx(z, dims);
            auto const Z = Eigen::VectorX<qpp::idx>::Map(z_vec.data(), z_vec.size());

            EXPECT_EQ(std::popcount(x & z), X.dot(Z));
        }
    }
}

//! @brief Figure 1.20 and equations 1.46 through 1.51
TEST(chapter1_4, deutsch_jozsa_algorithm)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto constexpr n = 7u;
    auto constexpr _2_pow_n = static_cast<qpp::idx>(std::pow(2, n));
    auto constexpr inv_sqrt_2_pow_n = 1. / std::sqrt(_2_pow_n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;
    auto const functions = std::array{ random_constant_function(n), random_balanced_function(n) };

    for(auto&& i : { 0, 1 })
    {
        auto const& f = functions[i];
        auto const constant = (i == 0);
        if (constant)
        {
            EXPECT_THAT((std::array{ 0u, 1u }), testing::Contains(f[0]));
            EXPECT_TRUE(std::all_of(policy, f.cbegin(), f.cend(), [&](auto&& x){ return x == f[0]; }));
        }
        else
        {
            EXPECT_TRUE(std::all_of(policy, f.cbegin(), f.cend(), [&](auto&& x){ return (x == 0u) || (x == 1u); }));
            EXPECT_EQ(std::reduce(policy, f.cbegin(), f.cend(), 0u, std::plus<>{}), f.size() / 2);
        }

        auto const Uf = matrixU_n(matrix_n(f));

        EXPECT_MATRIX_EQ(Uf * Uf.adjoint(), Eigen::MatrixXcd::Identity(2 * _2_pow_n, 2 * _2_pow_n));
        EXPECT_MATRIX_EQ(Uf, Uf.adjoint());

        auto const psi0 = qpp::kron(qpp::kronpow(0_ket, n), 1_ket);

        auto const psi1 = (qpp::kronpow(qpp::gt.H, n + 1) * psi0).eval();

        auto const expected_psi1 = qpp::kron(std::transform_reduce(policy, range.begin(), range.end()
            , Eigen::VectorXcd::Zero(_2_pow_n).eval()
            , std::plus<>{}
            , [&](auto&& i)
        {
            return Eigen::VectorXcd::Unit(_2_pow_n, i).eval();
        }), inv_sqrt_2_pow_n * inv_sqrt2 * (0_ket - 1_ket));
        EXPECT_MATRIX_CLOSE(psi1, expected_psi1, 1e-12);

        auto const psi2 = (Uf * psi1).eval();
        auto const expected_psi2 = qpp::kron(std::transform_reduce(policy, range.begin(), range.end()
            , Eigen::VectorXcd::Zero(_2_pow_n).eval()
            , std::plus<>{}
            , [&](auto&& i)
        {
            return (std::pow(-1, f[i]) * Eigen::VectorXcd::Unit(_2_pow_n, i)).eval();
        }), inv_sqrt_2_pow_n * inv_sqrt2 * (0_ket - 1_ket));
        EXPECT_MATRIX_CLOSE(psi2, expected_psi2, 1e-12);

        auto const dims = std::vector<qpp::idx>(n, 2u);
        auto const psi3 = (qpp::kron(qpp::kronpow(qpp::gt.H, n), qpp::gt.Id2) * psi2).eval();
        auto const expected_psi3 = qpp::kron(std::transform_reduce(policy, range.begin(), range.end()
            , Eigen::VectorXcd::Zero(_2_pow_n).eval()
            , std::plus<>{}
            , [&](auto&& x)
        {
            return std::transform_reduce(std::execution::seq, range.begin(), range.end()
                , Eigen::VectorXcd::Zero(_2_pow_n).eval()
                , std::plus<>{}
                , [&](auto&& z)
            {
                return (std::pow(-1, (std::popcount(x & z) + f[x]) % 2) * Eigen::VectorXcd::Unit(_2_pow_n, z)).eval();
            });
        }), 1. / _2_pow_n * inv_sqrt2 * (0_ket - 1_ket));
        EXPECT_MATRIX_CLOSE(psi3, expected_psi3, 1e-12);

        auto constexpr target = std::views::iota(0u, n) | std::views::common;
        auto const [result, probabilities, resulting_state] = qpp::measure(psi3, qpp::gt.Id(_2_pow_n), { target.begin(), target.end() });

        if (constant)
        {
            EXPECT_EQ(result, 0);
            EXPECT_NEAR(probabilities[0], 1., 1e-12);
        }
        else
        {
            EXPECT_NE(result, 0);
            EXPECT_NEAR(probabilities[0], 0., 1e-12);
        }

        if constexpr (print_text)
        {
            std::cerr << "-----------------------------\n";
            std::cerr << ">> f:\n" << Eigen::VectorX<qpp::idx>::Map(f.data(), f.size()) << '\n';
            std::cerr << ">> Uf:\n" << qpp::disp(Uf) << '\n';
            std::cerr << ">> psi0:\n" << qpp::disp(psi0) << '\n';
            std::cerr << ">> psi1:\n" << qpp::disp(psi1) << '\n';
            std::cerr << ">> psi2:\n" << qpp::disp(psi2) << '\n';
            std::cerr << ">> psi3:\n" << qpp::disp(psi3) << '\n';
            std::cerr << ">> Measurement result: " << result << '\n';
            std::cerr << ">> Probabilities: ";
            std::cerr << qpp::disp(probabilities, ", ") << '\n';
            std::cerr << ">> Resulting states:\n";
            for (auto&& it : resulting_state)
                std::cerr << qpp::disp(it) << "\n\n";
        }
    }
}
