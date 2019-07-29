#include "ActivationFunction/SoftmaxFunction.h"



#include <iostream>
// https://github.com/yixuan/MiniDNN/blob/master/include/Activation/Softmax.h
void
SoftmaxFunction::apply_function(Eigen::MatrixXf &input) const {
    // Subtract maximum of each column to lower numerical errors and apply exp
    auto output = (input.rowwise() - input.colwise().maxCoeff()).array().exp();

    // softmax: exp(a)/sum(exp(Matrix))
    input.array() = (output.rowwise() / output.colwise().sum());
}

Eigen::MatrixXf
SoftmaxFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                                  const Eigen::MatrixXf &dC_da) const {
    // Eigen::Array<float, 1, Eigen::Dynamic>
    // temp=softmax_input.cwiseProduct(input).colwise().sum();
    // output=softmax_input.array()*(input.array().rowwise() - temp);

    //std::cout << "input m_a\n" << m_a << "\ndC_da\n" << dC_da << std::endl;

    //auto F = dC_da;
    //auto A = m_a;

    Eigen::Array<float, 1, Eigen::Dynamic> a_dot_f =
            m_a.cwiseProduct(dC_da).colwise().sum();
    return (m_a.array() * (dC_da.array().rowwise() - a_dot_f)).matrix();
/*    std::cout << "Output min 1 with input\n" << out << std::endl;

    //return output;

    Eigen::Array<float, 1, Eigen::Dynamic> a_dot_f2 = A.cwiseProduct(F).colwise().sum();
    std::cout << "Output min 2 with input\n" << A.array() * (F.array().rowwise() - a_dot_f2) << std::endl;
    std::cout << "Min a_dot\n" << a_dot_f2 << std::endl;

    //http://saitcelebi.com/tut/output/part2.html


    Eigen::MatrixXf output_new(m_a.rows(),m_a.cols());
    output_new.setZero();
    auto row_ones=Eigen::RowVectorXf::Ones(m_a.rows());
    for (long i=0; i<m_a.cols(); i++) {
        Eigen::MatrixXf temp=(m_a.col(i)*row_ones).cwiseProduct(Eigen::MatrixXf::Identity(m_a.rows(),m_a.rows())-row_ones.transpose()*m_a.col(i).transpose());
        //std::cout << "tempMat\n" << temp << "\nm_a i col\n" << m_a.col(i) << std::endl;
        output_new.col(i)=temp*dC_da.col(i);
    }

    std::cout << "seitcelebi result with input\n" << output_new << std::endl;

    Eigen::MatrixXf output_new_new(m_a.rows(),m_a.cols());
    Eigen::MatrixXf temp(m_a.rows(),m_a.rows());
    temp.setZero();
    auto row_ones_new=Eigen::MatrixXf::Ones(m_a.cols(),m_a.rows());
    temp=(m_a*row_ones_new).cwiseProduct(Eigen::MatrixXf::Identity(m_a.rows(),m_a.rows())-row_ones_new.transpose()*m_a.transpose());
    std::cout << "seitcelebi 2 without input\n" << temp  << std::endl;
    output_new_new=temp*dC_da;

    std::cout << "seitcelebi result 2 with input\n" << output_new_new << std::endl;



    Eigen::MatrixXf output(m_a.rows(), m_a.cols());


    // TODO Test other code
    for (long i = 0; i < m_a.rows(); i++) {
        for (long j = 0; j < m_a.cols(); j++) {
            if (i == j) {
                output(i, j) = m_a(i) * (1 - m_a(i));
            }
            else {
                output(i, j) = -m_a(i) * m_a(j);
            }
        }
    }
    std::cout << "\nunsummed output for \n" << output  <<"\noutput for summed\n" << output.colwise().sum() << "\noutput for multiplied\n "<< output.colwise().sum()<< std::endl;*/
    //return out;

    /*
     *     # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix
     multiplication s = softmax.reshape(-1,1)
     return np.diagflat(s) - np.dot(s,s.T)
     *
     *
     * auto temp = np.diag(s)

      for i in range(len(jacobian_m)):
      for j in range(len(jacobian_m)):
      if i == j:
      jacobian_m[i][j] = s[i] * (1-s[i])
      else:
      jacobian_m[i][j] = -s[i]*s[j]
      return jacobian_m*/
}