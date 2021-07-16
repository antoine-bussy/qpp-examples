#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Equations 2.133 through 2.137, figure 2.3 and exercise 2.69
TEST(chapter2_3, superdense_coding)
{
    using namespace std::complex_literals;
    auto const psi = qpp::st.b00;
    auto const gates = std::vector<Eigen::Matrix2cd>{ qpp::gt.Id2, qpp::gt.Z, qpp::gt.X, 1i * qpp::gt.Y };
    auto const messages = std::vector<std::string>{ "00", "01", "10", "11" };
    auto const bell_basis = (Eigen::Matrix4cd{} << qpp::st.b00, qpp::st.b10, qpp::st.b01, qpp::st.b11).finished();

    EXPECT_MATRIX_CLOSE(bell_basis.adjoint() * bell_basis, Eigen::Matrix4cd::Identity(), 1e-12);

    if (print_text)
        std::cerr << "Bell basis:\n" << bell_basis << "\n\n";

    for (auto&& i : std::views::iota(0u, messages.size()))
    {
        auto const& sent_message = messages[i];
        auto const sent_state = qpp::apply(psi, gates[i], { 0 });

        auto const [result, probabilities, resulting_state] = qpp::measure(sent_state, bell_basis);
        auto const& received_message = messages[result];
        EXPECT_EQ(received_message, sent_message);
        if (print_text)
            std::cerr << "Sent message: " << sent_message << ", received: " << received_message << '\n';
    }
}
