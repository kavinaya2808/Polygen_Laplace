//=============================================================================
// Copyright 2024 Astrid Bunge, Sven Wagner, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include "../common_util.h"
#include <pmp/visualization/mesh_viewer.h>
#include "Surface/Smoothing.h"
#include <imgui.h>
#include "Surface/[AW11]Laplace.h"
#include "Surface/[dGBD20]Laplace.h"
#include "Surface/GeodesicsInHeat.h"
#include "Surface/Curvature.h"
#include "Surface/Parameterization.h"
#include "Surface/PolySimpleLaplace.h"
#include "Surface/PolySmoothing.h"
#include "Surface/SpectralProcessing.h"
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>

using namespace pmp;

#include <algorithm>
#include <tuple>
#include <numeric> // for iota

//=============================================================================

class Viewer : public MeshViewer
{
public:
    Viewer(const char* title, int width, int height);
    void load_mesh(const char* filename) override;

protected:
    void process_imgui() override;
    void draw(const std::string& _draw_mode) override;
    void color_code_condition_numbers(int laplace, int min_point);

private:
    Smoothing smoother_;
    double min_cond = -1, max_cond = -1;
    bool show_uv_layout_;
    int mesh_index_ = 0;

public:
    void apply_selective_trace_optimization(int laplace,
                                            int baseline_min_point,
                                            double fraction_to_optimize,
                                            bool compute_global_cond = true);

};


void Viewer::load_mesh(const char* filename)
{
    min_cond = -1;
    max_cond = -1;
    MeshViewer::load_mesh(filename);
}

Viewer::Viewer(const char* title, int width, int height)
    : MeshViewer(title, width, height), smoother_(mesh_)
{
    set_draw_mode("Hidden Line");
    crease_angle_ = 0.0;
    show_uv_layout_ = false;
}

void Viewer::process_imgui()
{
    if (ImGui::CollapsingHeader("Load Mesh", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const char* listbox_items[] = {"dual_Bunny.off", "fandisk.obj",
                                       "fertility.obj", "gear.obj",
                                       "horse.obj", "rockerArm.obj"};
        int listbox_item_current = mesh_index_;
        ImGui::ListBox(" ", &listbox_item_current, listbox_items,
                       IM_ARRAYSIZE(listbox_items), 5);
        if (listbox_item_current != mesh_index_ || mesh_.n_vertices() == 0)
        {
            std::stringstream ss;
            ss << DATA_PATH << listbox_items[listbox_item_current];

            load_mesh(ss.str().c_str());

            mesh_index_ = listbox_item_current;
        }
        ImGui::Spacing();
        ImGui::Spacing();
    }
    // initialize settings
    static int laplace = 0;
    static int min_point = 2;

    MeshViewer::process_imgui();

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Polygon Laplace",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::RadioButton("Alexa & Wardetzky Laplace", &laplace, 1);
        ImGui::RadioButton("deGoes Laplace", &laplace, 3);
        ImGui::RadioButton("Virtual Refinement Laplace", &laplace, 0);
        ImGui::RadioButton("Diamond Laplace", &laplace, 2);
        ImGui::RadioButton("Harmonic Laplace", &laplace, 4);

        ImGui::Spacing();
        if (laplace == 0 || laplace == 2)
        {
            ImGui::Text("Choose your minimizing point ");

            ImGui::Spacing();

            ImGui::RadioButton("Naïve (Centroid)", &min_point, 0);
            ImGui::RadioButton("Simple (Area Minimizer)", &min_point, 2);
            ImGui::RadioButton("Robust (Trace Minimizer)", &min_point, 3);

            ImGui::Spacing();
        }
        if (laplace == 1)
        {
            ImGui::PushItemWidth(100);
            ImGui::SliderFloat("Lambda", &poly_laplace_lambda_, 0.01, 3.0);
            ImGui::PopItemWidth();
        }
        else if (laplace == 3)
        {
            ImGui::PushItemWidth(100);
            ImGui::SliderFloat("Lambda", &deGoes_laplace_lambda_, 0.01, 3.0);
            ImGui::PopItemWidth();
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Make it robust",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(10);

        if (ImGui::Button("Mesh Optimization"))
        {
            SmoothingConfigs oConf(25, false, false, false, false);
            PolySmoothing polySmoothing(mesh_, oConf);
            polySmoothing.optimize(5);
            update_mesh();
        }

        if (ImGui::Button("Color Code Condition Number"))
        {
            color_code_condition_numbers(laplace, min_point);
            renderer_.set_specular(0);
            update_mesh();
            set_draw_mode("Hidden Line");
        }

        // inside ImGui::CollapsingHeader("Make it robust", ...)
        static float select_fraction = 0.05f; // default 5%
        ImGui::PushItemWidth(150);
        ImGui::SliderFloat("Fraction to optimize", &select_fraction, 0.0f, 1.0f, "%.2f");
        ImGui::Text("Selected fraction: %.0f%%", select_fraction * 100.0f);
        ImGui::PopItemWidth();
        if (ImGui::Button("Apply Selective Trace Optimization"))
        {
            apply_selective_trace_optimization(laplace, AreaMinimizer, select_fraction, true);
            renderer_.set_specular(0);
            // update_mesh() is called at the end of the function already, but safe to call again
            update_mesh();
            set_draw_mode("Hidden Line");
        }



        ImGui::Unindent(10);
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Applications", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(10);

        if (ImGui::Button("Mean Curvature"))
        {
            Curvature analyzer(mesh_, false);
            analyzer.visualize_curvature(laplace, min_point, true);
            renderer_.use_cold_warm_texture();
            update_mesh();
            set_draw_mode("Texture");
            show_uv_layout_ = false;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        static float timestep = 0.01;
        float lb = 0.001;
        float ub = 1.0;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("TimeStep", &timestep, lb, ub);
        ImGui::PopItemWidth();

        if (ImGui::Button("Smoothing"))
        {
            Scalar dt = timestep * radius_ * radius_;
            try
            {
                smoother_.implicit_smoothing(dt, laplace, min_point, true);
            }
            catch (const SolverException& e)
            {
                std::cerr << e.what() << std::endl;
            }

            update_mesh();
            set_draw_mode("Hidden Line");
            show_uv_layout_ = false;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Geodesics"))
        {
            GeodesicsInHeat heat(mesh_, laplace, min_point, false, false,
                                 DiffusionStep(2));
            Eigen::VectorXd dist, geodist;

            heat.compute_geodesics();
            heat.getDistance(0, dist, geodist);

            update_mesh();
            renderer_.use_checkerboard_texture();
            set_draw_mode("Texture");
            show_uv_layout_ = false;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Parametrization"))
        {
            try
            {
                Parameterization param(mesh_);
                param.harmonic(laplace, min_point);
            }
            catch (const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
            }

            update_mesh();
            renderer_.use_checkerboard_texture();
            set_draw_mode("Texture");
            show_uv_layout_ = true;
        }

        ImGui::Unindent(10);
    }
}
void Viewer::apply_selective_trace_optimization(int laplace,
                                                int baseline_min_point,
                                                double fraction_to_optimize,
                                                bool compute_global_cond)
{
    using Triplet = Eigen::Triplet<double>;

    int F = mesh_.n_faces();

    // get or create face properties
    auto face_impr = mesh_.face_property<double>("f:improvement");
    if (!face_impr)
        face_impr = mesh_.add_face_property<double>("f:improvement");

    auto face_color = mesh_.face_property<Color>("f:color");
    if (!face_color)
        face_color = mesh_.add_face_property<Color>("f:color");

    struct FaceInfo
    {
        Face f;
        double delta;
        Eigen::MatrixXd S_baseline;
        Eigen::MatrixXd S_trace;
        std::vector<int> vidx;
    };

    std::vector<FaceInfo> finfos;
    finfos.reserve(F);

    // 1) per-face baseline/trace stiffness and improvement
    for (auto f : mesh_.faces())
    {
        Eigen::MatrixXd poly;
        get_polygon_from_face(mesh_, f, poly); // repo helper

        // baseline weights
        Eigen::VectorXd w_baseline;
        if (baseline_min_point == Centroid_)
        {
            int val = (int)poly.rows();
            w_baseline = Eigen::VectorXd::Ones(val);
            w_baseline /= (double)val;
        }
        else
        {
            find_area_minimizer_weights(poly, w_baseline);
        }

        Eigen::Vector3d v_baseline = poly.transpose() * w_baseline;

        // baseline local stiffness
        Eigen::MatrixXd Si_baseline;
        localCotanMatrix(poly, v_baseline, w_baseline, Si_baseline);

        // trace-minimizer
        Eigen::VectorXd w_trace;
        find_trace_minimizer_weights(poly, w_trace);
        Eigen::Vector3d v_trace = poly.transpose() * w_trace;

        Eigen::MatrixXd Si_trace;
        localCotanMatrix(poly, v_trace, w_trace, Si_trace);

        double tr_baseline = Si_baseline.trace();
        double tr_trace = Si_trace.trace();
        double delta = (tr_baseline > 1e-18) ? (tr_baseline - tr_trace) / tr_baseline : 0.0;

        // collect vertex indices in face order
        std::vector<int> vidx;
        auto h0 = mesh_.halfedge(f);
        auto h = h0;
        do {
            vidx.push_back(mesh_.to_vertex(h).idx());
            h = mesh_.next_halfedge(h);
        } while (h != h0);

        finfos.push_back({f, delta, Si_baseline, Si_trace, vidx});
    }

    // check one sample face for debugging
    if (!finfos.empty()) {
        auto &fi = finfos[0];
        std::cout << "[DBG] sample face idx=" << fi.f.idx()
                  << " delta=" << fi.delta
                  << " tr_base=" << fi.S_baseline.trace()
                  << " tr_trace=" << fi.S_trace.trace() << std::endl;
    }

    // write improvement prop (and compute max for normalization)
    double max_impr = 0.0;
    for (auto &fi : finfos) {
        face_impr[fi.f] = fi.delta;
        max_impr = std::max(max_impr, fi.delta);
    }

    // sort faces by delta
    std::vector<int> idxs(finfos.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(),
                     [&](int a, int b){ return finfos[a].delta > finfos[b].delta; });

    int K = std::max(1, (int)std::round((double)finfos.size() * fraction_to_optimize));
    std::vector<char> use_trace_mask(finfos.size(), 0);
    for (int i = 0; i < K; ++i) use_trace_mask[idxs[i]] = 1;

    std::cout << "[STO] Faces total: " << finfos.size()
              << "  Selected K: " << K
              << " (fraction=" << fraction_to_optimize << ")\n";

    // color faces by improvement (normalized)
    for (size_t i = 0; i < finfos.size(); ++i)
    {
        double d = finfos[i].delta;
        double norm = (max_impr > 0.0) ? (d / max_impr) : 0.0;
        if (use_trace_mask[i])
            face_color[finfos[i].f] = Color(1.0, 0.0, 1.0); // magenta = selected
        else
            face_color[finfos[i].f] = Color(0.2 + 0.6*norm, 0.6 - 0.4*norm, 0.9 - 0.6*norm);
    }

    // assemble global stiffness
    int nVerts = mesh_.n_vertices();
    std::vector<Triplet> triplets;
    triplets.reserve(finfos.size() * 9);

    for (size_t i = 0; i < finfos.size(); ++i)
    {
        const FaceInfo& fi = finfos[i];
        const std::vector<int>& vids = fi.vidx;
        int n = (int)vids.size();
        const Eigen::MatrixXd& Slocal = use_trace_mask[i] ? fi.S_trace : fi.S_baseline;

        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
            {
                double v = Slocal(r, c);
                if (std::abs(v) > 1e-16)
                    triplets.emplace_back(vids[r], vids[c], v);
            }
    }

    Eigen::SparseMatrix<double> S_global(nVerts, nVerts);
    S_global.setFromTriplets(triplets.begin(), triplets.end());
    // enforce symmetry
    Eigen::SparseMatrix<double> S_sym = 0.5 * (S_global + Eigen::SparseMatrix<double>(S_global.transpose()));
    S_sym.makeCompressed();

    if (compute_global_cond)
    {
        bool firstEigZero = false;
        double cond_after = get_condition_number(-S_sym, firstEigZero);

        // baseline global
        std::vector<Triplet> triplets_base;
        for (auto &fi : finfos)
        {
            const std::vector<int>& vids = fi.vidx;
            int n = (int)vids.size();
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c)
                {
                    double vv = fi.S_baseline(r, c);
                    if (std::abs(vv) > 1e-16)
                        triplets_base.emplace_back(vids[r], vids[c], vv);
                }
        }
        Eigen::SparseMatrix<double> S_base(nVerts, nVerts);
        S_base.setFromTriplets(triplets_base.begin(), triplets_base.end());
        S_base.makeCompressed();
        Eigen::SparseMatrix<double> S_base_sym = 0.5 * (S_base + Eigen::SparseMatrix<double>(S_base.transpose()));
        S_base_sym.makeCompressed();

        double cond_base = get_condition_number(-S_base_sym, firstEigZero);

        // helper to query few extremal eigenvalues using Spectra
        auto print_extremal_eigs = [&](const Eigen::SparseMatrix<double>& A, const std::string& name){
            using OpType = Spectra::SparseSymMatProd<double>;
            OpType op(A);
            int nev = 3;
            int ncv = std::min((int)A.rows(), 6*nev);
            Spectra::SymEigsSolver<OpType> eigs(op, nev, ncv);
            eigs.init();
            eigs.compute(Spectra::SortRule::LargestAlge);
            Eigen::VectorXd largest = eigs.eigenvalues();
            std::cout << "[EIG] " << name << " largest: ";
            for (int i=0;i<largest.size();++i) std::cout << largest[i] << " ";
            std::cout << "\n";
        };

        print_extremal_eigs(S_base_sym, "S_base_sym");
        print_extremal_eigs(S_sym, "S_sym");

        std::cout << "Selective Trace Optimized faces: " << K << " / " << finfos.size() << std::endl;
        std::cout << "Global condition baseline: " << cond_base << std::endl;
        std::cout << "Global condition after selective trace: " << cond_after << std::endl;
        if (cond_base > 0.0)
            std::cout << "Relative improvement: " << (cond_base - cond_after) / cond_base * 100.0 << " %\n";
    }

    update_mesh();
    set_draw_mode("Hidden Line");
}



void Viewer::color_code_condition_numbers(int laplace, int min_point)
{
    auto face_color = mesh_.face_property<Color>("f:color");
    auto face_cond = mesh_.face_property<double>("f:condition");

    // compute global condition number
    Eigen::Vector3d values;
    double cond = condition_number(mesh_, laplace, min_point, values, false);
    std::cout << "Condition Number: " << cond << std::endl;

    // compute per-face condition numbers
    Eigen::MatrixXd Si;
    Eigen::VectorXd w;
    Eigen::Vector3d p;
    Eigen::MatrixXd poly;
    for (Face f : mesh_.faces())
    {
        // collect polygon vertices
        get_polygon_from_face(mesh_, f, poly);

        // compute weights for the polygon
        if (min_point == Centroid_)
        {
            int val = (int)poly.rows();
            w = Eigen::MatrixXd::Ones(val, 1);
            w /= (double)val;
        }
        else if (min_point == AreaMinimizer)
        {
            find_area_minimizer_weights(poly, w);
        }
        else
        {
            find_trace_minimizer_weights(poly, w);
        }
        Eigen::Vector3d min;

        // compute virtual vertex
        min = poly.transpose() * w;

        // get per-face Laplace matrix
        localCotanMatrix(poly, min, w, Si);

        // compute per-face condition number
        Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigs(Si);
        face_cond[f] =
            eigs.eigenvalues()[Si.rows() - 1] / eigs.eigenvalues()[1];
    }

    // compute max/min condition number for color coding
    if (max_cond == -1 && min_cond == -1)
    {
        std::vector<double> cond_numbers;
        for (auto f : mesh_.faces())
        {
            cond_numbers.push_back(face_cond[f]);
        }
        std::ranges::sort(cond_numbers);
        max_cond = cond_numbers[int(0.99 * mesh_.n_faces())];
        min_cond = cond_numbers[0];
    }

    // turn condition number into color
    for (auto f : mesh_.faces())
    {
        auto good_col = Color(0.39, 0.74, 1); // Turquoise (good)
        auto ok_col = Color(1, 0.74, 0);      // Orange (okay)
        auto bad_col = Color(1, 0.0, 1);      // Purple (bad)

        double blend = fmin(
            1.0, fmax(0.0, (face_cond[f] - min_cond) / (max_cond - min_cond)));
        face_color[f] = (blend < 0.5)
                            ? (1 - blend) * good_col + blend * ok_col
                            : (1 - blend) * ok_col + blend * bad_col;
    }
}

void Viewer::draw(const std::string& draw_mode)
{
    // draw the mesh
    renderer_.draw(projection_matrix_, modelview_matrix_, draw_mode);

    // draw uv layout
    if (draw_mode == "Texture" && show_uv_layout_)
    {
        // clear depth buffer
        glClear(GL_DEPTH_BUFFER_BIT);

        // setup viewport
        GLint size = std::min(width(), height()) / 4;
        glViewport(width() - size - 1, height() - size - 1, size, size);

        // setup matrices
        mat4 P = ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        mat4 M = mat4::identity();

        // draw mesh once more
        renderer_.draw(P, M, "Texture Layout");

        // reset viewport
        glViewport(0, 0, width(), height());
    }
}

int main(int argc, char** argv)
{
    Viewer window("Polygon Laplace Demo", 800, 600);
    if (argc == 2)
        window.load_mesh(argv[1]);
    return window.run();
}
