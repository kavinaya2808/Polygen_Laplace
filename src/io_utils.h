#pragma once
#include <Eigen/Sparse>
#include <string>

void writeSparseMatrixMarket(const Eigen::SparseMatrix<double> &A, const std::string &filename);