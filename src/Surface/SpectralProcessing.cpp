//=============================================================================
// Copyright 2023 Astrid Bunge, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include "../common_util.h"
#include "SpectralProcessing.h"
#include "LaplaceConstruction.h"
#include "[dGBD20]Laplace.h"
#include "HarmonicBasis2D.h"
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/Util/GEigsMode.h>
#include <iostream>
#include <Spectra/Util/SelectionRule.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include "Spectra/SymGEigsSolver.h"
#include <Spectra/SymEigsShiftSolver.h>
#include "Spectra/SymEigsSolver.h"
#include <iomanip>


//=============================================================================

using namespace pmp;
using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

//=============================================================================

double solve_eigenvalue_problem(SurfaceMesh& mesh, int laplace, int face_point,
                                const std::string& meshname)
{
    std::string filename;
    if (laplace == Diamond)
    {
        filename = "eigenvalues_[BBA21]_" + meshname + ".csv";
    }
    else if (laplace == AlexaWardetzkyLaplace)
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << poly_laplace_lambda_;
        std::string s = stream.str();
        filename = "eigenvalues_[AW11]_l=" + s + "_" + meshname + ".csv";
    }
    else if (laplace == deGoesLaplace)
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << deGoes_laplace_lambda_;
        std::string s = stream.str();
        filename = "eigenvalues_[dGBD20]_l=" + s + "_" + meshname + ".csv";
    }
    else if (laplace == PolySimpleLaplace)
    {
        filename = "eigenvalues_[BHKB20]_" + meshname + ".csv";
    }
    else if (laplace == Harmonic)
    {
        filename = "eigenvalues_[MKB08]_" + meshname + ".csv";
    }
    std::ofstream ev_file(filename);

    ev_file << "computed,analytic,offset" << std::endl;
    Eigen::SparseMatrix<double> M, S;
    double error = 0.0;
    if (laplace == Harmonic)
    {
        buildStiffnessAndMass2d(mesh, S, M);
        lump_matrix(M);
        S *= -1.0;
    }
    else
    {
        setup_stiffness_matrices(mesh, S, laplace, face_point);
        setup_mass_matrices(mesh, M, laplace, face_point);
    }
    // Construct matrix operation object using the wrapper class SparseGenMatProd

    int num_eval = 49;
    int converge_speed = 5 * num_eval;

    // S and M are sparse
    using OpType =
        Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = Spectra::SparseSymMatProd<double>;
    OpType op(S, M);
    BOpType Bop(M);

    // Construct generalized eigen solver object, seeking three generalized
    // eigenvalues that are closest to zero. This is equivalent to specifying
    // a shift sigma = 0.0 combined with the SortRule::LargestMagn selection rule

    Spectra::SymGEigsShiftSolver<OpType, BOpType,
                                 Spectra::GEigsMode::ShiftInvert>
        geigs(op, Bop, num_eval, converge_speed, 1e-8);
    geigs.init();
    geigs.compute(Spectra::SortRule::LargestMagn);
    std::cout << "compute & init" << std::endl;

    Eigen::VectorXd analytic, evalues;
    Eigen::MatrixXd evecs;
    // Retrieve results
    if (geigs.info() == Spectra::CompInfo::Successful)
    {
        evalues.resize(num_eval);
        analytic.resize(num_eval);
        evalues = geigs.eigenvalues();
    }
    else
    {
        std::cout << "Eigenvalue computation failed!\n" << std::endl;
    }

    analytic_eigenvalues_unitsphere(analytic, num_eval);

    for (int i = 1; i < evalues.size(); i++)
    {
        ev_file << evalues(i) << "," << analytic(i) << ","
                << evalues(i) - analytic(i) << std::endl;
        error += pow(evalues(i) - analytic(i), 2);
    }

    error = sqrt(error / (double)evalues.size());
    std::cout << "Root mean squared error: " << error << std::endl;
    ev_file.close();

    return error;
}

void analytic_eigenvalues_unitsphere(Eigen::VectorXd& eval, int n)
{
    eval.resize(n);
    int i = 1;
    int band = 1;
    eval(0) = 0.0;
    for (int k = 0; k < 10; k++)
    {
        for (int j = 0; j <= 2 * band; j++)
        {
            eval(i) = -(double)band * ((double)band + 1.0);
            i++;
            if (i == n)
            {
                break;
            }
        }
        if (i == n)
        {
            break;
        }
        band++;
    }
}
//----------------------------------------------------------------------------

double factorial(int n)
{
    if (n == 0)
        return 1.0;
    return (double)n * factorial(n - 1);
}

//----------------------------------------------------------------------------

double scale(int l, int m)
{
    double temp = ((2.0 * (double)l + 1.0) * factorial(l - m)) /
                  (4.0 * std::numbers::pi * factorial(l + m));
    return sqrt(temp);
}

//----------------------------------------------------------------------------

double legendre_Polynomial(int l, int m, double x)
{
    // evaluate an Associated Legendre Polynomial P(l,m,x) at x
    double pmm = 1.0;
    if (m > 0)
    {
        double somx2 = sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; i++)
        {
            pmm *= (-fact) * somx2;
            fact += 2.0;
        }
    }
    if (l == m)
        return pmm;
    double pmmp1 = x * (2.0 * (double)m + 1.0) * pmm;
    if (l == m + 1)
        return pmmp1;
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll)
    {
        pll = ((2.0 * (double)ll - 1.0) * x * pmmp1 -
               ((double)ll + (double)m - 1.0) * pmm) /
              ((double)ll - (double)m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

//----------------------------------------------------------------------------

double sphericalHarmonic(pmp::Point p, int l, int m)
{
    // l is the band, range [0..n]
    // m in the range [-l..l]
    // transform cartesian to spherical coordinates, assuming r = 1

    double phi = atan2(p[0], p[2]) + std::numbers::pi;
    double cos_theta = p[1] / norm(p);
    const double sqrt2 = sqrt(2.0);
    if (m == 0)
        return scale(l, 0) * legendre_Polynomial(l, m, cos_theta);
    else if (m > 0)
        return sqrt2 * scale(l, m) * cos((double)m * phi) *
               legendre_Polynomial(l, m, cos_theta);
    else
        return sqrt2 * scale(l, -m) * sin(-(double)m * phi) *
               legendre_Polynomial(l, -m, cos_theta);
}
//----------------------------------------------------------------------------

double rmse_sh(SurfaceMesh& mesh, int laplace, int min_point_, bool lumped)
{
    auto points = mesh.vertex_property<Point>("v:point");

    // comparing eigenvectors up to the 8th Band of legendre polynomials
    int band = 3;

    double error;
    double sum = 0.0;
    Eigen::VectorXd y(mesh.n_vertices());

    Eigen::SparseMatrix<double> S, M;
    setup_stiffness_matrices(mesh, S, laplace, min_point_);
    setup_mass_matrices(mesh, M, laplace, min_point_, lumped);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(M);
    solver.factorize(M);
    for (int l = 1; l <= band; l++)
    {
        double eval = -l * (l + 1);
        for (int m = -l; m <= l; m++)
        {
            for (auto v : mesh.vertices())
            {
                y(v.idx()) = sphericalHarmonic(points[v], l, m);
            }
            Eigen::MatrixXd X = solver.solve(S * y);
            error = (y - 1.0 / eval * X).transpose() * M * (y - 1.0 / eval * X);
            error = sqrt(error / double(mesh.n_vertices()));
            sum += error;
        }
    }

    if (laplace == AlexaWardetzkyLaplace)
    {
        std::cout << "Error SH band recreation  (AlexaWardetzky Laplace, l="
                  << poly_laplace_lambda_ << "): " << sum << std::endl;
    }
    else if (laplace == deGoesLaplace)
    {
        std::cout << "Error SH band recreation  (deGoes Laplace, l="
                  << deGoes_laplace_lambda_ << "): " << sum << std::endl;
    }
    else if (laplace == Harmonic)
    {
        std::cout << "Error SH band recreation  (Harmonic Laplace: " << sum
                  << std::endl;
    }
    else
    {
        if (laplace == Diamond)
        {
            std::cout << "Diamond Laplace: ";
        }
        else if (laplace == PolySimpleLaplace)
        {
            std::cout << "Polysimple Laplace: ";
        }
        if (min_point_ == Centroid_)
        {
            std::cout << "Error SH band recreation (centroid): " << sum
                      << std::endl;
        }
        else if (min_point_ == AreaMinimizer)
        {
            std::cout << "Error SH band recreation (area Minimizer): " << sum
                      << std::endl;
        }
        else
        {
            std::cout << "Error SH band recreation (trace minimizer): " << sum
                      << std::endl;
        }
    }
    return sum;
}

double condition_number(pmp::SurfaceMesh& mesh, int laplace, int minpoint,
                        Eigen::Vector3d& values, bool generalized)
{
    const int numVerts = mesh.n_vertices();
    Eigen::SparseMatrix<double> S, M;
    if (laplace == Harmonic)
    {
        buildStiffnessAndMass2d(mesh, S, M);
        lump_matrix(M);
        S *= -1.0;
    }
    else
    {
        setup_stiffness_matrices(mesh, S, laplace, minpoint);
        setup_mass_matrices(mesh, M, laplace, minpoint, true);
    }

    //slice matrices so that only rows and cols for inner vertices remain
    std::vector<int> innerVertIdxs;
    for (auto v : mesh.vertices())
    {
        if (!mesh.is_boundary(v))
        {
            innerVertIdxs.push_back(v.idx());
        }
    }
    int nInnerVertIdxs = innerVertIdxs.size();

    Eigen::SparseMatrix<double> S_in_in(nInnerVertIdxs, nInnerVertIdxs);
    Eigen::SparseMatrix<double> M_in_in(nInnerVertIdxs, nInnerVertIdxs);
    if (nInnerVertIdxs == numVerts)
    {
        S_in_in = S;
        M_in_in = M;
    }
    else
    {
        Eigen::SparseMatrix<double> S_columns(S.rows(), nInnerVertIdxs);
        Eigen::SparseMatrix<double> M_columns(M.rows(), nInnerVertIdxs);
        Eigen::SparseMatrix<double, Eigen::RowMajor> S_rows(nInnerVertIdxs,
                                                            nInnerVertIdxs);
        Eigen::SparseMatrix<double, Eigen::RowMajor> M_rows(nInnerVertIdxs,
                                                            nInnerVertIdxs);

        // process rows and columns separately for linear runtime
        for (int i = 0; i < nInnerVertIdxs; i++)
        {
            S_columns.col(i) = S.col(innerVertIdxs[i]);
            M_columns.col(i) = M.col(innerVertIdxs[i]);
        }
        for (int i = 0; i < nInnerVertIdxs; i++)
        {
            S_rows.row(i) = S_columns.row(innerVertIdxs[i]);
            M_rows.row(i) = M_columns.row(innerVertIdxs[i]);
        }
        S_in_in = S_rows;
        M_in_in = M_rows;
    }

    int numEigValues = 3;
    int convergenceSpeed = std::min(40 * numEigValues, (int)S_in_in.rows());

    Eigen::VectorXd eigValsMax;
    Eigen::VectorXd eigValsMin;
    if (generalized)
    {
        // Construct generalized eigen solver object, requesting the largest generalized eigenvalue
        Spectra::SparseSymMatProd<double> sOpMax(-S_in_in);
        Spectra::SparseCholesky<double> sBOpMax(M_in_in);
        Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>,
                                Spectra::SparseCholesky<double>,
                                Spectra::GEigsMode::Cholesky>
            eigSolverMax(sOpMax, sBOpMax, 1, convergenceSpeed);
        eigSolverMax.init();
        eigSolverMax.compute(Spectra::SortRule::LargestAlge);
        eigValsMax = eigSolverMax.eigenvalues();

        switch (eigSolverMax.info())
        {
            case Spectra::CompInfo::NotComputed:
                std::cout << "Max Eig: Not Computed" << std::endl;
                break;
            case Spectra::CompInfo::NotConverging:
                std::cout << "Max Eig: Not Converging" << std::endl;
                break;
            case Spectra::CompInfo::NumericalIssue:
                std::cout << "Max Eig: Numerical Issue" << std::endl;
                break;
            default:
                break;
        }

        // Construct generalized eigen solver object, seeking three generalized
        // eigenvalues that are closest to zero. This is equivalent to specifying
        // a shift sigma = 0.0 combined with the SortRule::LargestMagn selection rule
        Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse> sOpMin(
            -S_in_in, M_in_in);
        Spectra::SparseSymMatProd<double> sBOpMin(M_in_in);
        Spectra::SymGEigsShiftSolver<
            Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>,
            Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert>
            eigSolverMin(sOpMin, sBOpMin, numEigValues, convergenceSpeed, -0.1);
        eigSolverMin.init();
        eigSolverMin.compute(Spectra::SortRule::LargestMagn);
        eigValsMin = eigSolverMin.eigenvalues();

        switch (eigSolverMin.info())
        {
            case Spectra::CompInfo::NotComputed:
                std::cout << "Min Eig: Not Computed" << std::endl;
                break;
            case Spectra::CompInfo::NotConverging:
                std::cout << "Min Eig: Not Converging" << std::endl;
                break;
            case Spectra::CompInfo::NumericalIssue:
                std::cout << "Min Eig: Numerical Issue" << std::endl;
                break;
            default:
                break;
        }
    }
    else
    {
        // Max Eigenvalue solver
        Spectra::SparseSymMatProd<double> sOpMax(-S_in_in);
        Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigSolverMax(
            sOpMax, 1, convergenceSpeed);
        eigSolverMax.init();
        eigSolverMax.compute(Spectra::SortRule::LargestAlge);
        eigValsMax = eigSolverMax.eigenvalues();

        switch (eigSolverMax.info())
        {
            case Spectra::CompInfo::NotComputed:
                std::cout << "Not Computed" << std::endl;
                break;
            case Spectra::CompInfo::NotConverging:
                std::cout << "Not Converging" << std::endl;
                break;
            case Spectra::CompInfo::NumericalIssue:
                std::cout << "Numerical Issue" << std::endl;
                break;
            default:
                break;
        }

        // Min Eigenvalue solver
        Spectra::SparseSymShiftSolve<double> sOpMin(-S_in_in);
        Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>>
            eigSolverMin(sOpMin, numEigValues, convergenceSpeed, -0.1);
        eigSolverMin.init();
        eigSolverMin.compute(Spectra::SortRule::LargestMagn);
        eigValsMin = eigSolverMin.eigenvalues();

        switch (eigSolverMin.info())
        {
            case Spectra::CompInfo::NotComputed:
                std::cout << "Not Computed" << std::endl;
                break;
            case Spectra::CompInfo::NotConverging:
                std::cout << "Not Converging" << std::endl;
                break;
            case Spectra::CompInfo::NumericalIssue:
                std::cout << "Numerical Issue" << std::endl;
                break;
            default:
                break;
        }
    }

    values(0) = eigValsMax.coeff(0);
    values(1) =
        eigValsMin.coeff(numEigValues - 1 - (numVerts == innerVertIdxs.size()));
    values(2) = values(0) / values(1);
    return values(2);
}
//=============================================================================

double get_condition_number(const Eigen::SparseMatrix<double>& M,
                            bool firstEigZero)
{
    int numEigValues = 3;
    int convergenceSpeed = std::min(40 * numEigValues, (int)M.rows());

    Eigen::VectorXd eigValsMax;
    Eigen::VectorXd eigValsMin;

    Spectra::SparseSymMatProd<double> sOpMax(M);
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigSolverMax(
        sOpMax, 1, convergenceSpeed);
    eigSolverMax.init();
    eigSolverMax.compute(Spectra::SortRule::LargestAlge);
    eigValsMax = eigSolverMax.eigenvalues();

    switch (eigSolverMax.info())
    {
        case Spectra::CompInfo::NotComputed:
            std::cout << "Not Computed" << std::endl;
            break;
        case Spectra::CompInfo::NotConverging:
            std::cout << "Not Converging" << std::endl;
            break;
        case Spectra::CompInfo::NumericalIssue:
            std::cout << "Numerical Issue" << std::endl;
            break;
        default:
            break;
    }

    // Min Eigenvalue solver
    Spectra::SparseSymShiftSolve<double> sOpMin(M);
    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>>
        eigSolverMin(sOpMin, numEigValues, convergenceSpeed, -0.1);
    eigSolverMin.init();
    eigSolverMin.compute(Spectra::SortRule::LargestMagn);
    eigValsMin = eigSolverMin.eigenvalues();

    switch (eigSolverMin.info())
    {
        case Spectra::CompInfo::NotComputed:
            std::cout << "Not Computed" << std::endl;
            break;
        case Spectra::CompInfo::NotConverging:
            std::cout << "Not Converging" << std::endl;
            break;
        case Spectra::CompInfo::NumericalIssue:
            std::cout << "Numerical Issue" << std::endl;
            break;
        default:
            break;
    }

    //std::cout << eigValsMin << std::endl << eigValsMax << std::endl;
    std::cout << "Condition number of application matrix: "
              << eigValsMax.coeff(0) /
                     eigValsMin.coeff(numEigValues - 1 - (firstEigZero))
              << std::endl;
    return eigValsMax.coeff(0) /
           eigValsMin.coeff(numEigValues - 1 - (firstEigZero));
}


double compute_condition_number(const Eigen::SparseMatrix<double>& A_in)
{
    using OpType = Spectra::SparseSymMatProd<double>;

    const int n = (int)A_in.rows();
    if (n == 0 || A_in.cols() != A_in.rows())
        return std::numeric_limits<double>::infinity();

    Eigen::SparseMatrix<double> B = A_in;
    B.makeCompressed();

    // --- 1) compute largest algebraic eigenvalue ---
    auto compute_largest = [&](const Eigen::SparseMatrix<double>& M)->std::pair<double,bool>{
        OpType op(M);
        int nev = 1;
        int ncv = std::min((int)M.rows(), std::max(40, 8 * nev));
        Spectra::SymEigsSolver<OpType> eigs(op, nev, ncv);
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestAlge);
        if (eigs.info() != Spectra::CompInfo::Successful) return {0.0, false};
        return {eigs.eigenvalues()[0], true};
    };

    auto [lambda_max_raw, ok_max] = compute_largest(B);
    if (!ok_max) return std::numeric_limits<double>::infinity();

    // --- 2) compute several smallest algebraic eigenvalues (we'll use them to decide sign and pick min) ---
    int nev_min = std::min(5, std::max(1, (int)B.rows()));
    int ncv_min = std::min((int)B.rows(), std::max(20, 6 * nev_min));
    OpType op_min(B);
    Spectra::SymEigsSolver<OpType> eigs_min(op_min, nev_min, ncv_min);
    eigs_min.init();
    eigs_min.compute(Spectra::SortRule::SmallestAlge);

    std::vector<double> small_eigs;
    if (eigs_min.info() == Spectra::CompInfo::Successful) {
        Eigen::VectorXd vals = eigs_min.eigenvalues();
        small_eigs.resize(vals.size());
        for (int i=0;i<vals.size();++i) small_eigs[i] = vals[i];
    }

    // --- 3) decide sign robustly and flip B if it looks negative-semidefinite ---
    // Criteria:
    //  - lambda_max_raw is very small (near zero), *and*
    //  - we find negative small eigenvalues (algebraic min < -threshold)
    //
    // thresholds:
    double sign_eps_abs = 1e-12; // absolute threshold for lambda_max being "tiny"
    double neg_eig_threshold = 1e-8; // consider a small_eig << -neg_eig_threshold as clearly negative

    bool looks_negative_semidef = false;
    if (!small_eigs.empty()) {
        // if algebraic smallest is significantly negative and lambda_max is ~0 -> negative-semidef
        if (small_eigs[0] < -neg_eig_threshold && std::abs(lambda_max_raw) < sign_eps_abs) {
            looks_negative_semidef = true;
        }
        // also if many of the sampled small_eigs are negative -> negative
        int neg_count = 0;
        for (double v : small_eigs) if (v < -neg_eig_threshold) ++neg_count;
        if (neg_count >= (int)std::ceil(0.5 * (double)small_eigs.size())) looks_negative_semidef = true;
    } else {
        // if Spectra failed to compute small eigs but largest is extremely small, assume negative/degenerate
        if (std::abs(lambda_max_raw) < sign_eps_abs) looks_negative_semidef = true;
    }

    double lambda_max = lambda_max_raw;
    if (looks_negative_semidef || lambda_max_raw <= 0.0) {
        // flip sign and recompute lambda_max on flipped matrix
        B = -B;
        B.makeCompressed();
        std::tie(lambda_max, ok_max) = compute_largest(B);
        if (!ok_max) return std::numeric_limits<double>::infinity();
        if (lambda_max <= 0.0) {
            // degenerate after flipping too
            return std::numeric_limits<double>::infinity();
        }

        // recompute small_eigs on flipped matrix (optional but safer)
        OpType op_min2(B);
        Spectra::SymEigsSolver<OpType> eigs_min2(op_min2, nev_min, ncv_min);
        eigs_min2.init();
        eigs_min2.compute(Spectra::SortRule::SmallestAlge);
        small_eigs.clear();
        if (eigs_min2.info() == Spectra::CompInfo::Successful) {
            Eigen::VectorXd vals2 = eigs_min2.eigenvalues();
            small_eigs.resize(vals2.size());
            for (int i=0;i<vals2.size();++i) small_eigs[i] = vals2[i];
        }
    } else {
        // use lambda_max as computed
        lambda_max = lambda_max_raw;
    }

    // --- 4) pick a robust lambda_min (smallest positive above eps_pos), fallback to clamp ---
    double eps_pos = std::max(1e-14, std::abs(lambda_max) * 1e-13);
    double lambda_min = 0.0;
    bool found_pos = false;
    for (double ev : small_eigs) {
        if (ev > eps_pos) {
            if (!found_pos) { lambda_min = ev; found_pos = true; }
            else lambda_min = std::min(lambda_min, ev);
        }
    }

    if (!found_pos) {
        if (!small_eigs.empty()) {
            double algebraic_min = small_eigs[0];
            double tol = std::max(1e-14, std::abs(lambda_max) * 1e-12);
            if (algebraic_min > tol) lambda_min = algebraic_min;
            else lambda_min = tol;
        } else {
            lambda_min = std::max(1e-14, std::abs(lambda_max) * 1e-12);
        }
    }

    double cond = std::abs(lambda_max) / std::abs(lambda_min);

    if (cond > 1e10 || cond < 1.0) { // print only extremely suspicious cases
        std::ostringstream oss;
        oss << "[CONDDBG] n=" << n
            << " lambda_max=" << lambda_max
            << " small_eigs:";
        for (double v : small_eigs) oss << " " << v;
        oss << " chosen_lambda_min=" << lambda_min
            << " eps_pos=" << eps_pos
            << " cond=" << cond << "\n";
        std::cerr << oss.str();
    }

    if (!std::isfinite(cond)) return std::numeric_limits<double>::infinity();
    return cond;
}


std::tuple<double,double,double> compute_generalized_spectral_metrics(
    const Eigen::SparseMatrix<double>& S_in,
    const Eigen::SparseMatrix<double>& M_in)
{
    Eigen::SparseMatrix<double> S = S_in; S.makeCompressed();
    Eigen::SparseMatrix<double> M = M_in; M.makeCompressed();

    int n = (int)S.rows();
    if (n == 0 || M.rows() != n || M.cols() != n) {
        return {std::nan(""), std::nan(""), std::numeric_limits<double>::infinity()};
    }

    try {
        // 1) try a safe (regularized) approach to estimate lam_max:
        double lam_max = std::nan("");
        // try generalized cholesky-based largest eigenvalue if M is reasonably positive-definite
        try {
            Spectra::SparseSymMatProd<double> sOp(-S);
            Spectra::SparseCholesky<double> sB(M);
            int nev_max = 1;
            int ncv_max = std::min(n, std::max(40, 8 * nev_max));
            Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>,
                                     Spectra::SparseCholesky<double>,
                                     Spectra::GEigsMode::Cholesky>
                geigs_max(sOp, sB, nev_max, ncv_max);
            geigs_max.init();
            geigs_max.compute(Spectra::SortRule::LargestAlge);
            if (geigs_max.info() == Spectra::CompInfo::Successful) lam_max = geigs_max.eigenvalues()[0];
        } catch (...) {
            // fallback to a power-type iteration (solve M z = S x)
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solverM;
            solverM.compute(M);
            if (solverM.info() == Eigen::Success) {
                Eigen::VectorXd x = Eigen::VectorXd::Random(n); x.normalize();
                double last_lambda = 0.0;
                for (int it = 0; it < 200; ++it) {
                    Eigen::VectorXd y = S * x;
                    Eigen::VectorXd z = solverM.solve(y);
                    if (solverM.info() != Eigen::Success) break;
                    double lambda = x.dot(z);
                    double normz = z.norm();
                    if (normz == 0.0) break;
                    x = z / normz;
                    if (it > 5 && std::abs(lambda - last_lambda) / (1.0 + std::abs(last_lambda)) < 1e-12) {
                        lam_max = std::abs(lambda);
                        break;
                    }
                    last_lambda = lambda;
                }
                if (!std::isfinite(lam_max)) lam_max = std::abs(last_lambda);
            }
        }

        // second fallback: if lam_max still NaN, try simple Spectra on -S alone
        if (!std::isfinite(lam_max)) {
            try {
                Spectra::SparseSymMatProd<double> opS(-S);
                int nev = 1;
                int ncv = std::min(n, std::max(40, 8 * nev));
                Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(opS, nev, ncv);
                eigs.init();
                eigs.compute(Spectra::SortRule::LargestAlge);
                if (eigs.info() == Spectra::CompInfo::Successful) lam_max = eigs.eigenvalues()[0];
            } catch (...) {}
        }

        // 2) attempt to compute smallest non-zero generalized eigenvalue via shift-invert
        double lam_min_nonzero = std::nan("");
        std::vector<double> reg_eps = {0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8};
        for (double eps : reg_eps) {
            try {
                Eigen::SparseMatrix<double> Mreg = M;
                if (eps > 0.0) {
                    for (int k=0;k<Mreg.rows();++k) Mreg.coeffRef(k,k) += eps;
                    Mreg.makeCompressed();
                }

                Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse> shiftOp(-S, Mreg);
                Spectra::SparseSymMatProd<double> bOp(Mreg);

                int nev_min = std::min(6, std::max(2, n/10));
                int ncv_min = std::min(n, std::max(20, 6 * nev_min));
                int kreq = std::min(nev_min, std::max(1, n-1));

                Spectra::SymGEigsShiftSolver< Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>,
                                              Spectra::SparseSymMatProd<double>,
                                              Spectra::GEigsMode::ShiftInvert >
                    geigs_min(shiftOp, bOp, kreq, ncv_min, 0.0);
                geigs_min.init();
                geigs_min.compute(Spectra::SortRule::LargestMagn);
                if (geigs_min.info() == Spectra::CompInfo::Successful) {
                    Eigen::VectorXd vals = geigs_min.eigenvalues();
                    double eps_rel = std::max(1e-14, (std::isfinite(lam_max) ? std::abs(lam_max) * 1e-12 : 1e-14));
                    std::vector<double> nonzeros;
                    for (int i=0;i<vals.size();++i) if (std::abs(vals[i]) > eps_rel) nonzeros.push_back(std::abs(vals[i]));
                    if (!nonzeros.empty()) {
                        std::sort(nonzeros.begin(), nonzeros.end());
                        lam_min_nonzero = nonzeros.front();
                        break;
                    } else if (vals.size() > 0) {
                        lam_min_nonzero = std::abs(vals[0]);
                        break;
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "[compute_generalized_spectral_metrics] Spectra attempt eps=" << eps << " exception: " << e.what() << std::endl;
                // keep trying other eps
            } catch (...) {
                // keep trying
            }
        }

        double kappa = std::numeric_limits<double>::infinity();
        if (std::isfinite(lam_max) && std::isfinite(lam_min_nonzero) && lam_min_nonzero > 0.0) {
            kappa = std::abs(lam_max / lam_min_nonzero);
        }

        return {lam_min_nonzero, lam_max, kappa};
    } catch (const std::exception &e) {
        std::cerr << "[compute_generalized_spectral_metrics] exception: " << e.what() << std::endl;
        return {std::nan(""), std::nan(""), std::numeric_limits<double>::infinity()};
    }
}