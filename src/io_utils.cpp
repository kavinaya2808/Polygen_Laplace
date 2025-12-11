#include "io_utils.h"
#include <fstream>
#include <iostream>
#include <iomanip>

void writeSparseMatrixMarket(const Eigen::SparseMatrix<double> &A, const std::string &filename)
{
    std::ofstream out(filename);
    if(!out) {
        std::cerr << "writeSparseMatrixMarket: cannot open " << filename << "\n";
        return;
    }
    out << "%%MatrixMarket matrix coordinate real general\n";
    out << A.rows() << " " << A.cols() << " " << A.nonZeros() << "\n";
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            out << (it.row() + 1) << " " << (it.col() + 1) << " " << std::setprecision(16) << it.value() << "\n";
        }
    }
    out.close();
}