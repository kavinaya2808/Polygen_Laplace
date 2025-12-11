// polydemo.cpp
//=============================================================================
// Copyright 2024 Astrid Bunge, Sven Wagner, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include <algorithm>
#include <tuple>
#include <numeric>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <regex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <limits>

#include "io_utils.h"

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

#include <Eigen/IterativeLinearSolvers>

using namespace pmp;

// small helper to convert pmp::Point to Eigen::Vector3d (safe)
static Eigen::Vector3d to_vec3(const pmp::Point &p) {
    return Eigen::Vector3d(static_cast<double>(p[0]),
                           static_cast<double>(p[1]),
                           static_cast<double>(p[2]));
}

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


    void run_full_evaluation_and_save(int laplace, int baseline_min_point,
                                      int steps = 21, int sh_band = 3, int sh_m = 1);


public:

    struct FaceInfo {
        Face f;
        double delta;
        Eigen::MatrixXd S_baseline;
        Eigen::MatrixXd S_trace;
        std::vector<int> vidx;
        // virtual vertex positions (world coords)
        Eigen::Vector3d v_baseline = Eigen::Vector3d::Zero();
        Eigen::Vector3d v_trace    = Eigen::Vector3d::Zero();
        bool local_valid = true;
    };
    std::vector<FaceInfo> face_infos_;   // cached per-face info

    // dashboard vectors
    std::vector<double> dash_fractions_;
    std::vector<double> dash_cond_base_;
    std::vector<double> dash_cond_after_;

    std::atomic<bool> eval_running_{false};

    // helpers
    void compute_face_info(int baseline_min_point);
    std::pair<double,double> assemble_and_compute_cond(const std::vector<char>& use_trace_mask);
    void run_selective_sweep(int steps, int baseline_min_point);


    bool show_virtual_vertices_ = false;
    int  virtual_vertex_mode_ = 0;
    float virtual_point_size_ = 6.0f;
    bool show_only_selected_face_vv_ = false;


    std::string mesh_filename_;

    std::mutex dash_mutex_;

    double cond_area_cached_  = std::numeric_limits<double>::quiet_NaN();
    double cond_trace_cached_ = std::numeric_limits<double>::quiet_NaN();

private:
    GLuint vv_vao_ = 0;
    GLuint vv_points_vbo_ = 0;
    GLuint vv_lines_vbo_ = 0;
    GLuint vv_prog_ = 0;
    int    vv_point_count_ = 0;
    int    vv_line_vertex_count_ = 0;
    bool   vv_gl_initialized_ = false;

    void init_virtual_vv_gl();
    void update_virtual_vv_buffers();
    void render_virtual_vertices(); 


    std::pair<int,double> run_cg_count(const Eigen::SparseMatrix<double>& A,
                                       const Eigen::VectorXd& b,
                                       Eigen::VectorXd& x,
                                       double tol = 1e-8, int maxit = 20000);


    void assemble_S_mixed_and_mass(const std::vector<char>& use_trace_mask,
                                   Eigen::SparseMatrix<double>& S_mixed_sym,
                                   Eigen::SparseMatrix<double>& M_out,
                                   int laplace, int baseline_min_point);


    double compute_poisson_L2_error(const Eigen::SparseMatrix<double>& S_mixed_sym,
                                    const Eigen::SparseMatrix<double>& M,
                                    int sh_l, int sh_m,
                                    double &out_cg_time,
                                    double &out_iterations,
                                    int &out_iters);
                                    


    void append_csv_line(const std::string &fname, const std::string &line);


    void write_svg_two_series(const std::string &fname,
                              const std::vector<double> &x,
                              const std::vector<double> &y1,
                              const std::vector<double> &y2,
                              const std::string &label1,
                              const std::string &label2,
                              const std::string &title);


    void write_svg_single_series(const std::string &fname,
                                 const std::vector<double> &x,
                                 const std::vector<double> &y,
                                 const std::string &label,
                                 const std::string &title);


    void run_experiments(const std::string &output_prefix,
                         const std::vector<std::string> &sto_strategies = {"trace_top","cond_top","curvature_top","random"},
                         const std::vector<double> &fractions = {0.01, 0.05, 0.10, 0.25, 0.5, 0.75},
                         const std::vector<int> &taus = {5,3,-1},
                         int numIters = 100,
                         int use_laplace = 0,
                         int use_baseline_min_point = 2);

}; // end class Viewer


static double my_triangle_area(const pmp::Point &a, const pmp::Point &b, const pmp::Point &c) {
    Eigen::Vector3d A = to_vec3(a);
    Eigen::Vector3d B = to_vec3(b);
    Eigen::Vector3d C = to_vec3(c);
    return 0.5 * ((B - A).cross(C - A)).norm();
}
static double safe_positive_tol(double reference_val) {
    double base = 1e-14;
    if (std::isfinite(reference_val) && std::abs(reference_val) > 0.0)
        base = std::max(base, std::abs(reference_val) * 1e-12);
    return base;
}

static bool sparse_matrix_has_nonfinite(const Eigen::SparseMatrix<double>& A) {
    for (int k = 0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
            if (!std::isfinite(it.value())) return true;
    return false;
}

static double min_triangle_angle_deg(const pmp::Point &a, const pmp::Point &b, const pmp::Point &c) {
    Eigen::Vector3d A = (to_vec3(b) - to_vec3(a)).normalized();
    Eigen::Vector3d B = (to_vec3(c) - to_vec3(b)).normalized();
    Eigen::Vector3d C = (to_vec3(a) - to_vec3(c)).normalized();
    // angles at vertices
    double angA = std::acos(fmax(-1.0, fmin(1.0, A.dot(-C)))) * 180.0 / M_PI;
    double angB = std::acos(fmax(-1.0, fmin(1.0, B.dot(-A)))) * 180.0 / M_PI;
    double angC = std::acos(fmax(-1.0, fmin(1.0, C.dot(-B)))) * 180.0 / M_PI;
    return std::min({angA, angB, angC});
}

static int count_flipped_faces(pmp::SurfaceMesh &mesh, const std::vector<double> &area_before) {
    int flips = 0;
    size_t idx = 0;
    for (auto f : mesh.faces()) {
        // collect vertices of face
        auto h0 = mesh.halfedge(f);
        auto h = h0;
        std::vector<pmp::Point> pts;
        do {
            pts.push_back(mesh.position(mesh.to_vertex(h)));
            h = mesh.next_halfedge(h);
        } while (h != h0);

        if (pts.size() >= 3) {
            Eigen::Vector3d v0 = to_vec3(pts[0]);
            Eigen::Vector3d v1 = to_vec3(pts[1]);
            Eigen::Vector3d v2 = to_vec3(pts[2]);
            double area_signed = 0.5 * ((v1 - v0).cross(v2 - v0))[2]; // Z-component proxy
            double before = (idx < area_before.size()) ? area_before[idx] : std::abs(area_signed);
            if (area_before.size() && (area_signed * before < 0.0)) flips++;
        }
        idx++;
    }
    return flips;
}


static void compute_geometry_stats(pmp::SurfaceMesh &mesh,
                                   const std::vector<pmp::Point> &pos_before,
                                   double &out_avg_disp, double &out_max_disp,
                                   double &out_mean_area_rel_change, double &out_max_area_rel_change,
                                   double &out_min_triangle_angle_deg,
                                   double &out_percent_faces_below_10deg,
                                   int &out_num_flipped)
{
    int V = mesh.n_vertices();
    // vertex displacement
    out_avg_disp = 0.0;
    out_max_disp = 0.0;
    int cntV = 0;
    for (auto v : mesh.vertices()) {
        pmp::Point p = mesh.position(v);
        if ((int)v.idx() < (int)pos_before.size()) {
            double d = (to_vec3(p) - to_vec3(pos_before[v.idx()])).norm();
            out_avg_disp += d;
            out_max_disp = std::max(out_max_disp, d);
            cntV++;
        }
    }
    if (cntV) out_avg_disp /= double(cntV);

    // face area changes and angle stats
    double sum_rel_area = 0.0;
    double max_rel_area = 0.0;
    double min_angle = std::numeric_limits<double>::infinity();
    int faces_below_10 = 0;
    int total_faces = 0;
    std::vector<double> before_areas;
    before_areas.reserve(mesh.n_faces());
    for (auto f : mesh.faces()) {
        // gather face vertices
        std::vector<pmp::Point> pts;
        auto h0 = mesh.halfedge(f);
        auto h = h0;
        do {
            pts.push_back(mesh.position(mesh.to_vertex(h)));
            h = mesh.next_halfedge(h);
        } while (h != h0);

        if (pts.size() >= 3) {
            double A = 0.0;
            // sum triangle fan areas
            for (size_t i=1;i+1<pts.size();++i) A += my_triangle_area(pts[0], pts[i], pts[i+1]);
            before_areas.push_back(A);
            // compute angles per triangle fan and take min
            double local_min = std::numeric_limits<double>::infinity();
            for (size_t i=0;i+1<pts.size();++i) {
                double ang = min_triangle_angle_deg(pts[0], pts[i], pts[i+1]);
                local_min = std::min(local_min, ang);
            }
            if (local_min < min_angle) min_angle = local_min;
            if (local_min < 10.0) faces_below_10++;
            total_faces++;
        }
    }

    // For area changes we need positions before per-face — we do a simple approximation using pos_before
    int fi = 0;
    for (auto f : mesh.faces()) {
        std::vector<pmp::Point> pts;
        auto h0 = mesh.halfedge(f);
        auto h = h0;
        do {
            pts.push_back(mesh.position(mesh.to_vertex(h)));
            h = mesh.next_halfedge(h);
        } while (h != h0);
        // current area
        if (pts.size() >= 3) {
            double Acur = 0.0;
            for (size_t i=1;i+1<pts.size();++i) Acur += my_triangle_area(pts[0], pts[i], pts[i+1]);
            // compute previous positions for same indices:
            std::vector<pmp::Point> pts_before;
            h = h0;
            do {
                int vid = mesh.to_vertex(h).idx();
                if (vid < (int)pos_before.size()) pts_before.push_back(pos_before[vid]);
                else pts_before.push_back(mesh.position(mesh.to_vertex(h))); // fallback
                h = mesh.next_halfedge(h);
            } while (h != h0);
            double Abefore = 0.0;
            if (pts_before.size() >= 3) {
                for (size_t i=1;i+1<pts_before.size();++i) Abefore += my_triangle_area(pts_before[0], pts_before[i], pts_before[i+1]);
            } else Abefore = Acur;
            double rel = (Abefore > 1e-15) ? std::abs((Acur - Abefore) / Abefore) : std::abs(Acur - Abefore);
            sum_rel_area += rel;
            max_rel_area = std::max(max_rel_area, rel);
        }
        fi++;
    }

    out_mean_area_rel_change = (total_faces>0) ? (sum_rel_area / double(total_faces)) : 0.0;
    out_max_area_rel_change = max_rel_area;
    out_min_triangle_angle_deg = (min_angle==std::numeric_limits<double>::infinity()) ? 0.0 : min_angle;
    out_percent_faces_below_10deg = (total_faces>0) ? (100.0 * double(faces_below_10) / double(total_faces)) : 0.0;

    // Build before area list similarly:
    std::vector<double> areas_before;
    areas_before.reserve(mesh.n_faces());
    for (auto f : mesh.faces()) {
        auto h0 = mesh.halfedge(f);
        auto h = h0;
        std::vector<pmp::Point> pts_before;
        do {
            int vid = mesh.to_vertex(h).idx();
            if (vid < (int)pos_before.size()) pts_before.push_back(pos_before[vid]);
            else pts_before.push_back(mesh.position(mesh.to_vertex(h)));
            h = mesh.next_halfedge(h);
        } while (h != h0);
        double Abefore = 0.0;
        if (pts_before.size()>=3) {
            for (size_t i=1;i+1<pts_before.size();++i) Abefore += my_triangle_area(pts_before[0], pts_before[i], pts_before[i+1]);
        }
        areas_before.push_back(Abefore);
    }
    out_num_flipped = count_flipped_faces(mesh, areas_before);
}

static std::pair<double,double> assemble_cond_from_faceinfos(
    const std::vector<Viewer::FaceInfo>& local_face_infos,
    const std::vector<char>& use_trace_mask);

static GLuint compile_shader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024];
        glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
        std::cerr << "Shader compile error: " << buf << std::endl;
    }
    return s;
}

static std::string sanitize_mesh_name(const std::string &full)
{
    if (full.empty()) return std::string("mesh_eval");
    // strip directories
    size_t p = full.find_last_of("/\\");
    std::string s = (p == std::string::npos) ? full : full.substr(p + 1);
    // strip extension
    size_t dot = s.find_last_of('.');
    if (dot != std::string::npos) s = s.substr(0, dot);
    // replace non-alnum with underscore (safe filenames)
    s = std::regex_replace(s, std::regex(R"([^A-Za-z0-9_\-])"), "_");
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024];
        glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
        std::cerr << "Program link error: " << buf << std::endl;
    }
    return p;
}

//=============================================================================

void Viewer::load_mesh(const char* filename)
{
    mesh_filename_ = (filename ? std::string(filename) : std::string());
    min_cond = -1;
    max_cond = -1;
    MeshViewer::load_mesh(filename);

    compute_face_info(AreaMinimizer);
    init_virtual_vv_gl();
}

Viewer::Viewer(const char* title, int width, int height)
    : MeshViewer(title, width, height), smoother_(mesh_)
{
    set_draw_mode("Hidden Line");
    crease_angle_ = 0.0;
    show_uv_layout_ = false;
}

void Viewer::init_virtual_vv_gl()
{
    if (vv_gl_initialized_) return;

    // vertex shader: position + color; uses uniform MVP and point size
    const char* vs = R"GLSL(
    #version 330 core
    layout(location = 0) in vec3 in_pos;
    layout(location = 1) in vec3 in_col;
    uniform mat4 uMVP;
    uniform float uPointSize;
    out vec3 vColor;
    void main() {
        gl_Position = uMVP * vec4(in_pos, 1.0);
        gl_PointSize = uPointSize;
        vColor = in_col;
    }
    )GLSL";

    const char* fs = R"GLSL(
    #version 330 core
    in vec3 vColor;
    out vec4 fragColor;
    void main() {
        fragColor = vec4(vColor, 1.0);
    }
    )GLSL";

    GLuint sv = compile_shader(GL_VERTEX_SHADER, vs);
    GLuint sf = compile_shader(GL_FRAGMENT_SHADER, fs);
    vv_prog_ = link_program(sv, sf);
    glDeleteShader(sv);
    glDeleteShader(sf);

    // Create VAO + VBOs
    glGenVertexArrays(1, &vv_vao_);
    glGenBuffers(1, &vv_points_vbo_);
    glGenBuffers(1, &vv_lines_vbo_);

    // one VAO, we bind appropriate VBO before draw
    glBindVertexArray(vv_vao_);

    // points VBO layout (vec3 pos + vec3 color)
    glBindBuffer(GL_ARRAY_BUFFER, vv_points_vbo_);
    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)0);
    glEnableVertexAttribArray(1); // color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)(sizeof(float)*3));

    // lines VBO uses same layout; we'll bind it when drawing lines
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    vv_gl_initialized_ = true;
}

void Viewer::process_imgui()
{
    static bool show_dashboard = true;
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
    static float select_fraction = 0.05f; // default 5%
    static bool use_STO_mode = true;

    MeshViewer::process_imgui();

    ImGui::Spacing(); ImGui::Spacing(); ImGui::Spacing();

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

    ImGui::Spacing(); ImGui::Spacing(); ImGui::Spacing();

    if (ImGui::CollapsingHeader("Make it robust",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(10);
        if (ImGui::Button("Mesh Optimization"))
        {
            // read UI options
            double fraction_to_optimize = select_fraction;
            std::string strategy = "trace_top"; // or obtain from UI

            // compute face infos with AreaMinimizer baseline (so v_baseline exists)
            compute_face_info(AreaMinimizer);

            // build use_trace_mask as you do in apply_selective_trace_optimization
            size_t F = face_infos_.size();
            std::vector<int> idxs(F);
            std::iota(idxs.begin(), idxs.end(), 0);
            if (strategy == "trace_top") {
                std::stable_sort(idxs.begin(), idxs.end(), [&](int a,int b){ return face_infos_[a].delta > face_infos_[b].delta; });
            } else {
                std::mt19937 rng(123456);
                std::shuffle(idxs.begin(), idxs.end(), rng);
            }
            int K = std::max(1, (int)std::round(fraction_to_optimize * double(F)));
            std::vector<char> use_trace_mask(F, 0);
            for (int i = 0; i < K; ++i) use_trace_mask[idxs[i]] = 1;

            // set faceVirtuals on the mesh_ explicitly
            auto faceVirtuals = mesh_.face_property<Eigen::Vector3d>("f:Virtuals");
            if (!faceVirtuals) faceVirtuals = mesh_.add_face_property<Eigen::Vector3d>("f:Virtuals");
            int count_trace = 0;
            for (size_t i = 0; i < face_infos_.size(); ++i) {
                const auto &fi = face_infos_[i];
                Eigen::Vector3d chosen = use_trace_mask[i] ? fi.v_trace : fi.v_baseline;
                faceVirtuals[fi.f] = chosen;
                if (use_trace_mask[i]) ++count_trace;
            }

            std::cout << "[GUI] STO mask prepared. trace faces: " << count_trace << " / " << F << std::endl;

            // color faces for feedback (optional)
            auto face_color = mesh_.face_property<Color>("f:color");
            if (!face_color) face_color = mesh_.add_face_property<Color>("f:color");
            for (size_t i = 0; i < face_infos_.size(); ++i)
                face_color[face_infos_[i].f] = use_trace_mask[i] ? Color(1.0,0.0,1.0) : Color(0.8f,0.8f,0.8f);

            update_mesh();

            // configure smoothing to *keep* these faceVirtuals
            SmoothingConfigs oConf(25, /*fixBoundary=*/true, /*updateQuadrics=*/true, /*withCnum=*/false, /*generalizedCnum=*/false, /*lockFaceVirtuals=*/true);

            // construct poly smoother: it will NOT recompute virtuals because lockFaceVirtuals==true
            PolySmoothing polySmoothing(mesh_, oConf);
            try {
                polySmoothing.optimize(/*quadricsTau=*/5);
            } catch (const std::exception &e) {
                std::cerr << "Mesh optimization error: " << e.what() << std::endl;
            }

            update_mesh();
        }

        if (ImGui::Button("Color Code Condition Number"))
        {
            color_code_condition_numbers(laplace, min_point);
            renderer_.set_specular(0);
            update_mesh();
            set_draw_mode("Hidden Line");
        }

        ImGui::PushItemWidth(150);
        ImGui::SliderFloat("Fraction to optimize", &select_fraction, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Use STO for Mesh Optimization", &use_STO_mode);
        ImGui::Text("Selected fraction: %.0f%%", select_fraction * 100.0f);
        ImGui::PopItemWidth();
        if (ImGui::Button("Apply Selective Trace Optimization"))
        {
            if (select_fraction > 1.0f) select_fraction = 1.0f;
            if (select_fraction < 0.0f) select_fraction = 0.0f;

            apply_selective_trace_optimization(laplace, AreaMinimizer, select_fraction, true);
            renderer_.set_specular(0);
            update_mesh();
            set_draw_mode("Hidden Line");
        }

        // Background CSV worker: light / heavy evaluation
        ImGui::Separator();
        ImGui::Text("Evaluation (background workers)");

        if (ImGui::Button("Run CSVs (light)"))
        {
            if (eval_running_.load()) {
                std::cout << "[FullEval] already running, ignoring CSV request\n";
            } else {
                eval_running_.store(true);
                std::thread([this]() {
                    try {
                        std::vector<Viewer::FaceInfo> local_face_infos = this->face_infos_;
                        std::string mesh_name = "mesh_eval";
                        if (!this->mesh_filename_.empty()) {
                            mesh_name = sanitize_mesh_name(this->mesh_filename_);
                        }

                        const int steps = 21;
                        size_t F = local_face_infos.size();
                        std::vector<int> idxs(F);
                        std::iota(idxs.begin(), idxs.end(), 0);
                        std::stable_sort(idxs.begin(), idxs.end(), [&](int a,int b){
                            return local_face_infos[a].delta > local_face_infos[b].delta;
                        });

                        std::string csvname = mesh_name + "_full_eval.csv";
                        std::ofstream csv(csvname);
                        csv << "fraction,cond_base,cond_after\n";

                        for (int s = 0; s < steps; ++s) {
                            double frac = (steps==1)?1.0: double(s) / double(steps-1);
                            int K = std::max(0, (int)std::round(frac * double(F)));
                            std::vector<char> mask(F, 0);
                            for (int i=0;i<K;++i) mask[idxs[i]] = 1;
                            auto [cond_base, cond_after] = assemble_cond_from_faceinfos(local_face_infos, mask);
                            csv << std::setprecision(12) << frac << "," << cond_base << "," << cond_after << "\n";
                        }
                        csv.close();
                        std::cout << "[FullEval] CSV saved: " << csvname << "\n";
                    } catch (const std::exception &e) {
                        std::cerr << "[FullEval] CSV worker exception: " << e.what() << std::endl;
                    } catch (...) {
                        std::cerr << "[FullEval] CSV worker unknown exception\n";
                    }
                    eval_running_.store(false);
                }).detach();
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(creates one sto CSV)");

        if (ImGui::Button("Run Full Eval (heavy)"))
        {
            if (eval_running_.load()) {
                std::cout << "[FullEval] already running, ignoring full-eval request\n";
            } else {
                eval_running_.store(true);

                // capture local UI variables (make plain local copies)
                int use_laplace   = laplace;
                int use_min_point = min_point;
                int use_steps     = 21; // or expose as UI control
                int use_sh_band   = 3;
                int use_sh_m      = 1;

                // MUST compute face_infos_ on the main thread before launching the worker
                // (compute_face_info touches mesh/GL/read-only props)
                compute_face_info(use_min_point);

                // spawn worker thread, capturing use_* by value
                std::thread([this, use_laplace, use_min_point, use_steps, use_sh_band, use_sh_m]() {
                    try {
                        this->run_full_evaluation_and_save(use_laplace, use_min_point, use_steps, use_sh_band, use_sh_m);
                    } catch (const std::exception &e) {
                        std::cerr << "[FullEval-worker] exception: " << e.what() << std::endl;
                    } catch (...) {
                        std::cerr << "[FullEval-worker] unknown exception\n";
                    }
                    // mark finished so UI button can be used again
                    eval_running_.store(false);
                }).detach();
            }
        }

        static bool show_dashboard_local = true;
        ImGui::Checkbox("Show Dashboard Window", &show_dashboard_local);
        // ---------- Virtual vertex UI ----------
        ImGui::Separator();
        ImGui::Text("Virtual Vertices");
        ImGui::Checkbox("Show virtual vertices", &show_virtual_vertices_);
        ImGui::SameLine();
        ImGui::Checkbox("Only selected face", &show_only_selected_face_vv_); // requires per-face selection logic to be implemented

        ImGui::Text("Show mode:");
        ImGui::RadioButton("Baseline", &virtual_vertex_mode_, 0); ImGui::SameLine();
        ImGui::RadioButton("Trace", &virtual_vertex_mode_, 1); ImGui::SameLine();
        ImGui::RadioButton("Both", &virtual_vertex_mode_, 2);

        ImGui::SliderFloat("Point size", &virtual_point_size_, 1.0f, 12.0f);
        ImGui::Separator();

    }

    ImGui::Spacing(); ImGui::Spacing(); ImGui::Spacing();

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
        float LB = 0.001;
        float UB = 1.0;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("TimeStep", &timestep, LB, UB);
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

    if (show_dashboard)
    {
        ImGui::Begin("Selective Trace Dashboard"); // creates new floating window

        ImGui::Text("Fraction: %.1f%%", select_fraction * 100.0f);

        // show cached pure-case conds (if available)
        if (std::isfinite(cond_area_cached_)) {
            ImGui::Text("Cond (area-minimizer baseline): %.6g", cond_area_cached_);
            ImGui::SameLine();
        }
        if (std::isfinite(cond_trace_cached_)) {
            ImGui::Text("Cond (trace-minimizer full trace): %.6g", cond_trace_cached_);
        }

        if (ImGui::Button("Apply Current Fraction")) {
            compute_face_info(AreaMinimizer);
            size_t F = face_infos_.size();
            std::vector<int> idxs(F);
            std::iota(idxs.begin(), idxs.end(), 0);
            std::stable_sort(idxs.begin(), idxs.end(), [&](int a,int b){
                return face_infos_[a].delta > face_infos_[b].delta;
            });
            int K = std::max(0, (int)std::round(select_fraction * double(F)));
            std::vector<char> mask(F,0);
            for (int i=0;i<K;++i) mask[idxs[i]] = 1;

            auto face_color = mesh_.face_property<Color>("f:color");
            if (!face_color) face_color = mesh_.add_face_property<Color>("f:color");
            double max_impr = 0.0;
            for (auto &fi : face_infos_) max_impr = std::max(max_impr, fi.delta);
            for (size_t i=0;i<face_infos_.size();++i) {
                if (mask[i]) face_color[face_infos_[i].f] = Color(1.0,0.0,1.0);
                else {
                    double norm = (max_impr>0.0) ? (face_infos_[i].delta / max_impr) : 0.0;
                    face_color[face_infos_[i].f] = Color(0.2 + 0.6*norm, 0.6 - 0.4*norm, 0.9 - 0.6*norm);
                }
            }
            update_mesh();
            set_draw_mode("Hidden Line");
        }

        static int mark_face_idx = 0;
        ImGui::InputInt("Face idx to toggle", &mark_face_idx);
        if (ImGui::Button("Toggle face VV")) {
            auto showprop = mesh_.face_property<char>("f:show_vv");
            if (!showprop) showprop = mesh_.add_face_property<char>("f:show_vv");
            Face f = Face(mark_face_idx);
            if (f.is_valid()) {
                showprop[f] = !showprop[f];
                update_mesh();
            }
        }

        // ---------------- Run sweep button (non-blocking) ----------------
        static int sweep_steps = 10;
        ImGui::InputInt("Steps", &sweep_steps);
        if (sweep_steps < 1) sweep_steps = 1;

        if (ImGui::Button("Run sweep")) {
            compute_face_info(AreaMinimizer);
            int steps_to_run = sweep_steps;
            std::thread([this, steps_to_run](){
                this->run_selective_sweep(steps_to_run, AreaMinimizer);
            }).detach();
        }

        // --- When drawing the plot, read the dash vectors under lock ---
        {
            std::lock_guard<std::mutex> lock(dash_mutex_);
            if (!dash_fractions_.empty())
            {
                const size_t N = dash_fractions_.size();
                std::vector<double> base(dash_cond_base_.begin(), dash_cond_base_.end());
                std::vector<double> after(dash_cond_after_.begin(), dash_cond_after_.end());

                double min_val = std::numeric_limits<double>::infinity();
                double max_val = -std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < N; ++i)
                {
                    min_val = std::min(min_val, std::min(base[i], after[i]));
                    max_val = std::max(max_val, std::max(base[i], after[i]));
                }

                if (!std::isfinite(min_val) || !std::isfinite(max_val) || fabs(max_val - min_val) < 1e-30)
                {
                    min_val = 0.0;
                    max_val = 1.0;
                }

                std::vector<float> plot_base(N), plot_after(N);
                for (size_t i = 0; i < N; ++i)
                {
                    plot_base[i] = static_cast<float>((base[i] - min_val) / (max_val - min_val));
                    plot_after[i] = static_cast<float>((after[i] - min_val) / (max_val - min_val));
                }

                ImGui::Text("Condition number trend (normalized)");
                ImGui::Columns(2, "plot_cols", false);
                ImGui::SetColumnWidth(0, 70); // left column for Y labels

                {
                    std::ostringstream smax; smax << std::scientific << std::setprecision(2) << max_val;
                    ImGui::Text("%s", smax.str().c_str());
                    ImGui::Dummy(ImVec2(0, 40));
                    double mid_val = 0.5 * (max_val + min_val);
                    std::ostringstream smid; smid << std::scientific << std::setprecision(2) << mid_val;
                    ImGui::Text("%s", smid.str().c_str());
                    ImGui::Dummy(ImVec2(0, 40));
                    std::ostringstream smin; smin << std::scientific << std::setprecision(2) << min_val;
                    ImGui::Text("%s", smin.str().c_str());
                }

                ImGui::NextColumn();

                ImGui::PushItemWidth(300);
                ImVec2 plot_size = ImVec2(300, 120);
                ImGui::PlotLines("Baseline (normalized)", plot_base.data(), (int)N, 0, nullptr, 0.0f, 1.0f, plot_size);
                ImGui::PlotLines("Selective (normalized)", plot_after.data(), (int)N, 0, nullptr, 0.0f, 1.0f, plot_size);
                ImGui::PopItemWidth();

                ImGui::Columns(1);

                ImGui::Spacing();
                ImGui::Indent(70);
                {
                    std::array<float,5> ticks = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
                    for (size_t t = 0; t < ticks.size(); ++t)
                    {
                        if (t > 0) ImGui::SameLine();
                        std::ostringstream tx; tx << int(ticks[t]*100.0f) << "%";
                        ImGui::Text("%s", tx.str().c_str());
                    }
                }
                ImGui::Unindent();

                ImGui::Spacing();
                ImGui::Text("Example points:");
                std::ostringstream cur_base, cur_after;
                cur_base << "Baseline (first) = " << std::scientific << std::setprecision(2) << base.front();
                cur_after << "Selective (first) = " << std::scientific << std::setprecision(2) << after.front();
                ImGui::Text("%s", cur_base.str().c_str());
                ImGui::SameLine();
                ImGui::Text("%s", cur_after.str().c_str());
            }
        } // end dash lock scope

        if (ImGui::Button("Save CSV")) {
            std::ofstream csv("sto_results.csv");
            csv << "fraction,cond_base,cond_after\n";
            {
                std::lock_guard<std::mutex> lock(dash_mutex_);
                for (size_t i=0;i<dash_fractions_.size();++i)
                    csv << dash_fractions_[i] << "," << dash_cond_base_[i] << "," << dash_cond_after_[i] << "\n";
            }
            csv.close();
            std::cout << "Saved sto_results.csv\n";
        }

        ImGui::End(); // end floating window
    }

} // end process_imgui

// --- apply_selective_trace_optimization implementation ---
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

        FaceInfo fi;
        fi.f = f;
        fi.delta = delta;
        fi.S_baseline = Si_baseline;
        fi.S_trace = Si_trace;
        fi.vidx = vidx;
        fi.v_baseline = v_baseline;
        fi.v_trace = v_trace;
        finfos.push_back(fi);
    }

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
                if (r == c || std::abs(v) > 1e-18)
                    triplets.emplace_back(vids[r], vids[c], v);
            }
    }

    Eigen::SparseMatrix<double> S_global(nVerts, nVerts);
    S_global.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SparseMatrix<double> S_sym = 0.5 * (S_global + Eigen::SparseMatrix<double>(S_global.transpose()));
    S_sym.makeCompressed();

    if (compute_global_cond)
    {
        Eigen::SparseMatrix<double> S_neg = -S_sym;
        S_neg.makeCompressed();
        double cond_after = compute_condition_number(S_neg);

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

        Eigen::SparseMatrix<double> S_base_neg = -S_base_sym;
        S_base_neg.makeCompressed();
        double cond_base  = compute_condition_number(S_base_neg);

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


static std::pair<double,double> assemble_cond_from_faceinfos(
    const std::vector<Viewer::FaceInfo>& local_face_infos,
    const std::vector<char>& use_trace_mask)
{
    using Triplet = Eigen::Triplet<double>;
    int nVerts = 0;
    for (const auto &fi : local_face_infos) {
        for (int vid : fi.vidx) nVerts = std::max(nVerts, vid + 1);
    }
    if (nVerts == 0) return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};

    std::vector<Triplet> mixed_triplets;
    std::vector<Triplet> base_triplets;
    mixed_triplets.reserve(local_face_infos.size()*9);
    base_triplets.reserve(local_face_infos.size()*9);

    for (size_t i = 0; i < local_face_infos.size(); ++i)
    {
        const auto &fi = local_face_infos[i];
        int n = (int)fi.vidx.size();
        const Eigen::MatrixXd &Sbase = fi.S_baseline;
        const Eigen::MatrixXd &Strace = fi.S_trace;
        const Eigen::MatrixXd &Slocal = (use_trace_mask.size() ? (use_trace_mask[i] ? Strace : Sbase) : Sbase);
        for (int r=0;r<n;++r) for (int c=0;c<n;++c)
        {
            double vM = Slocal(r,c);
            if (std::abs(vM) > 1e-16) mixed_triplets.emplace_back(fi.vidx[r], fi.vidx[c], vM);
            double vB = Sbase(r,c);
            if (std::abs(vB) > 1e-16) base_triplets.emplace_back(fi.vidx[r], fi.vidx[c], vB);
        }
    }

    Eigen::SparseMatrix<double> S_mixed(nVerts, nVerts), S_base(nVerts, nVerts);
    S_mixed.setFromTriplets(mixed_triplets.begin(), mixed_triplets.end());
    S_mixed.makeCompressed();
    S_base.setFromTriplets(base_triplets.begin(), base_triplets.end());
    S_base.makeCompressed();

    Eigen::SparseMatrix<double> S_mixed_sym = 0.5 * (S_mixed + Eigen::SparseMatrix<double>(S_mixed.transpose()));
    S_mixed_sym.makeCompressed();
    Eigen::SparseMatrix<double> S_base_sym = 0.5 * (S_base + Eigen::SparseMatrix<double>(S_base.transpose()));
    S_base_sym.makeCompressed();

    Eigen::SparseMatrix<double> neg_mixed = -S_mixed_sym; neg_mixed.makeCompressed();
    Eigen::SparseMatrix<double> neg_base  = -S_base_sym;  neg_base.makeCompressed();

    double cond_after = compute_condition_number(neg_mixed);
    double cond_base  = compute_condition_number(neg_base);

    return {cond_base, cond_after};
}


// -----------------------------------------------------------------------------
// run_cg_count: run conjugate gradient on A x = b and return (iterations, time_sec)
// -----------------------------------------------------------------------------

std::pair<int,double> Viewer::run_cg_count(const Eigen::SparseMatrix<double>& A_in,
                                           const Eigen::VectorXd& b_in,
                                           Eigen::VectorXd& x,
                                           double tol, int maxit)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // Work on copies since we may flip sign
    Eigen::SparseMatrix<double> A = A_in;
    Eigen::VectorXd b = b_in;
    bool flipped = false;

    // Quick cheap SPD check using LLT
    auto is_spd = [&](const Eigen::SparseMatrix<double>& M)->bool {
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
        llt.compute(M);
        return llt.info() == Eigen::Success;
    };

    if (!is_spd(A)) {
        // try -A
        Eigen::SparseMatrix<double> negA = -A;
        if (is_spd(negA)) {
            A = negA;
            b = -b;      // solve (-A) x = -b => same x
            flipped = true;
        } else {
            std::cerr << "[CG DBG] Warning: matrix A is not SPD and -A is also not SPD. "
                      << "CG (SPD) may not converge; proceeding anyway.\n";
            // proceed with original A (may still work with preconditioner / CG)
        }
    }

    // Construct CG with IncompleteCholesky preconditioner type (as before).
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                             Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteCholesky<double>> cg;

    cg.setTolerance(tol);
    cg.setMaxIterations(maxit);

    // compute (this also prepares the default preconditioner internally)
    cg.compute(A);

    // initialize x to zeros (or preserve caller-provided initial guess if desired)
    x = Eigen::VectorXd::Zero(A.rows());

    // Solve
    Eigen::VectorXd x_sol = cg.solveWithGuess(b, x); // uses initial guess and returns solution
    x = x_sol;

    int iters = cg.iterations();
    auto info = cg.info();

    // compute residual w.r.t original system A_in * x = b_in for diagnostics
    Eigen::VectorXd r = A_in * x - b_in;
    double rel_res = (b_in.norm() > 0.0) ? (r.norm() / b_in.norm()) : r.norm();

    auto t1 = Clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    if (info != Eigen::Success) {
        std::cerr << "[CG DBG] CG info() != Success ("
                  << (int)info << "). iters=" << iters
                  << " rel_res=" << rel_res
                  << " time_s=" << secs << "\n";
    } else if (rel_res > std::max(1e-12, tol*10.0)) {
        // warn if residual larger than expected (loose threshold, keeps logs lean)
        std::cerr << "[CG DBG] CG finished iters=" << iters
                  << " rel_res=" << rel_res
                  << " time_s=" << secs << "\n";
    }

    return {iters, secs};
}

// -----------------------------------------------------------------------------
// append_csv_line: append a line into fname (create file if missing)
// -----------------------------------------------------------------------------
void Viewer::append_csv_line(const std::string &fname, const std::string &line)
{
    std::ofstream ofs(fname, std::ios::app);
    ofs << line << "\n";
    ofs.close();
}

// -----------------------------------------------------------------------------
// assemble_S_mixed_and_mass: assemble S_mixed_sym from face_infos_ and
// call setup_mass_matrices to create M_out.
// -----------------------------------------------------------------------------
void Viewer::assemble_S_mixed_and_mass(const std::vector<char>& use_trace_mask,
                                       Eigen::SparseMatrix<double>& S_mixed_sym,
                                       Eigen::SparseMatrix<double>& M_out,
                                       int laplace, int baseline_min_point)
{
    using Triplet = Eigen::Triplet<double>;
    int nVerts = mesh_.n_vertices();
    std::vector<Triplet> mixed_triplets;
    mixed_triplets.reserve(face_infos_.size()*9);

    for (size_t i = 0; i < face_infos_.size(); ++i)
    {
        const auto &fi = face_infos_[i];
        int n = (int)fi.vidx.size();
        const Eigen::MatrixXd &Sbase = fi.S_baseline;
        const Eigen::MatrixXd &Strace = fi.S_trace;
        const Eigen::MatrixXd &Slocal = (use_trace_mask.size() ? (use_trace_mask[i] ? Strace : Sbase) : Sbase);
        for (int r=0;r<n;++r) for (int c=0;c<n;++c)
        {
            double vM = Slocal(r,c);
            if (std::abs(vM) > 1e-16) mixed_triplets.emplace_back(fi.vidx[r], fi.vidx[c], vM);
        }
    }

    Eigen::SparseMatrix<double> S_mixed(nVerts, nVerts);
    S_mixed.setFromTriplets(mixed_triplets.begin(), mixed_triplets.end());
    S_mixed.makeCompressed();
    S_mixed_sym = 0.5 * (S_mixed + Eigen::SparseMatrix<double>(S_mixed.transpose()));
    S_mixed_sym.makeCompressed();

    // construct mass matrix using existing helper (mass does not depend on selection mask)
    Eigen::SparseMatrix<double> Mtmp;
    setup_mass_matrices(mesh_, Mtmp, laplace, baseline_min_point);
    M_out = Mtmp;
    M_out.makeCompressed();
}

// -----------------------------------------------------------------------------
// compute_poisson_L2_error: solve S u = M * (l*(l+1) * u_true) for spherical harmonic
// returns L2 error (mass-weighted) and outputs CG stats
// -----------------------------------------------------------------------------
double Viewer::compute_poisson_L2_error(const Eigen::SparseMatrix<double>& S_mixed_sym,
                                        const Eigen::SparseMatrix<double>& M,
                                        int sh_l, int sh_m,
                                        double &out_cg_time,
                                        double &out_iterations,
                                        int &out_iters)
{
    // --- collect inner + boundary vertex lists (global indices) ---
    std::vector<int> innerVertIdxs;
    std::vector<int> boundaryVertIdxs;
    innerVertIdxs.reserve(mesh_.n_vertices());
    boundaryVertIdxs.reserve(mesh_.n_vertices());
    for (auto v : mesh_.vertices()) {
        if (!mesh_.is_boundary(v)) innerVertIdxs.push_back(v.idx());
        else boundaryVertIdxs.push_back(v.idx());
    }
    int nInner = (int)innerVertIdxs.size();
    int nVerts = (int)S_mixed_sym.rows();
    if (nInner == 0 || nVerts == 0) {
        std::cerr << "[POISSON] No inner vertices or empty matrix\n";
        return std::numeric_limits<double>::infinity();
    }

    // --- mapping global -> local for inner vertices ---
    std::vector<int> g2l(nVerts, -1);
    for (int i = 0; i < nInner; ++i) g2l[innerVertIdxs[i]] = i;

    // --- compute u_true for ALL vertices (normalize to unit sphere first) ---
    Eigen::VectorXd u_true_full(nVerts);
    auto points = mesh_.vertex_property<pmp::Point>("v:point");
    if (!points) {
        std::cerr << "[POISSON] vertex property v:point missing\n";
        return std::numeric_limits<double>::infinity();
    }
    for (int vid = 0; vid < nVerts; ++vid) {
        // guard: some meshes may have non-contiguous ids, but earlier code assumes 0..nVerts-1
        pmp::Vertex vv(vid);
        if (!vv.is_valid()) {
            // fallback: set zero
            u_true_full(vid) = 0.0;
            continue;
        }
        pmp::Point p = points[vv];
        Eigen::Vector3d pp(p[0], p[1], p[2]);
        if (pp.norm() > 0.0) pp.normalize();
        // sphericalHarmonic signature in your code expects pmp::Point or Eigen::Vector3d.
        // adjust call if your function uses a different type.
        double val = sphericalHarmonic(pp, sh_l, sh_m);
        u_true_full(vid) = val;
    }

    // --- Build S_ii and M_ii triplets, and accumulate S_ib*u_b and M_ib*(coef*u_b) terms ---
    std::vector<Eigen::Triplet<double>> Strip, Mtrip;
    Strip.reserve(S_mixed_sym.nonZeros() / 4 + 10);
    Mtrip.reserve(M.nonZeros() / 4 + 10);

    Eigen::VectorXd rhs_from_S_ib = Eigen::VectorXd::Zero(nInner);   // S_ib * u_b (will be subtracted)
    Eigen::VectorXd rhs_from_M_ib = Eigen::VectorXd::Zero(nInner);   // M_ib * (coef * u_b) (will be added)

    double coef = double(sh_l) * double(sh_l + 1);

    for (int k = 0; k < S_mixed_sym.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(S_mixed_sym, k); it; ++it) {
            int gi = it.row();
            int gj = it.col();
            double v = it.value();
            int li = (gi >= 0 && gi < nVerts) ? g2l[gi] : -1;
            int lj = (gj >= 0 && gj < nVerts) ? g2l[gj] : -1;

            if (li != -1 && lj != -1) {
                // inner-inner entry
                Strip.emplace_back(li, lj, v);
            } else if (li != -1 && lj == -1) {
                // inner row, boundary column => contributes S_ib * u_b (move to rhs with minus sign)
                rhs_from_S_ib(li) += v * u_true_full(gj);
            } else if (li == -1 && lj != -1) {
                // boundary row, inner column -> when forming S_ii * u_i this doesn't enter directly,
                // because S_ib is inner row, boundary col. This case is handled via symmetry when iterating full matrix.
                // nothing to do here.
            }
        }
    }

    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            int gi = it.row();
            int gj = it.col();
            double v = it.value();
            int li = (gi >= 0 && gi < nVerts) ? g2l[gi] : -1;
            int lj = (gj >= 0 && gj < nVerts) ? g2l[gj] : -1;

            if (li != -1 && lj != -1) {
                // inner-inner mass
                Mtrip.emplace_back(li, lj, v);
            } else if (li != -1 && lj == -1) {
                // inner row, boundary col => contributes M_ib * f_b where f_b = coef * u_true_b
                rhs_from_M_ib(li) += v * (coef * u_true_full(gj));
            } else if (li == -1 && lj != -1) {
                // boundary row, inner col -> nothing extra for interior eq
            }
        }
    }

    Eigen::SparseMatrix<double> S_in_in(nInner, nInner), M_in_in(nInner, nInner);
    S_in_in.setFromTriplets(Strip.begin(), Strip.end());
    S_in_in.makeCompressed();
    M_in_in.setFromTriplets(Mtrip.begin(), Mtrip.end());
    M_in_in.makeCompressed();

    // --- build u_true inner vector ---
    Eigen::VectorXd u_true_in(nInner);
    for (int i = 0; i < nInner; ++i) u_true_in(i) = u_true_full(innerVertIdxs[i]);

    // --- assemble RHS: M_ii * (coef * u_true_i) + M_ib*(coef*u_b) - S_ib*u_b ---
    Eigen::VectorXd rhs = M_in_in * (coef * u_true_in);
    rhs += rhs_from_M_ib;
    rhs -= rhs_from_S_ib;

    // --- If S_in_in is numerically bad, apply tiny diagonal regularization (only if needed) ---
    {
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt_check;
        llt_check.compute(S_in_in);
        if (llt_check.info() != Eigen::Success) {
            // compute diag max magnitude
            double diag_max = 0.0;
            for (int k = 0; k < S_in_in.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(S_in_in, k); it; ++it)
                    if (it.row() == it.col()) diag_max = std::max(diag_max, std::abs(it.value()));
            double eps = std::max(1e-16, diag_max * 1e-12);
            std::vector<Eigen::Triplet<double>> add;
            add.reserve(nInner);
            for (int i = 0; i < nInner; ++i) add.emplace_back(i, i, eps);
            // merge: keep original triplets (Strip) + diag add
            std::vector<Eigen::Triplet<double>> all;
            all.reserve(Strip.size() + add.size());
            all.insert(all.end(), Strip.begin(), Strip.end());
            all.insert(all.end(), add.begin(), add.end());
            Eigen::SparseMatrix<double> S_reg(nInner, nInner);
            S_reg.setFromTriplets(all.begin(), all.end());
            S_reg.makeCompressed();
            S_in_in.swap(S_reg);
            std::cerr << "[POISSON] Regularized S_in_in with eps=" << eps << " (diag_max=" << diag_max << ")\n";
        }
    }

    // --- solve S_in_in * u_sol = rhs using your CG benchmark function ---
    Eigen::VectorXd u_sol(nInner);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto [iters, cgtime] = run_cg_count(S_in_in, rhs, u_sol, 1e-10, 20000);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // if CG didn't converge, report large error but still compute residual-based error if possible
    if (iters >= 20000) {
        std::cerr << "[POISSON] CG hit max iterations while solving Poisson\n";
    }

    // --- compute M-weighted L2 error on interior: sqrt( (u_sol-u_true_in)^T * M_in_in * (u_sol-u_true_in) ) ---
    Eigen::VectorXd diff = u_sol - u_true_in;
    Eigen::VectorXd Mdiff = M_in_in * diff;
    double err2 = diff.dot(Mdiff);
    if (!std::isfinite(err2) || err2 < 0.0) {
        if (err2 < 0.0 && err2 > -1e-12) err2 = 0.0; // clamp tiny negative due to roundoff
        else {
            std::cerr << "[POISSON] computed invalid err2 = " << err2 << "\n";
            out_cg_time = secs;
            out_iterations = double(iters);
            out_iters = iters;
            return std::numeric_limits<double>::infinity();
        }
    }
    double err = std::sqrt(std::max(0.0, err2));

    // --- fill outputs & return ---
    out_cg_time = secs;
    out_iterations = double(iters);
    out_iters = iters;
    return err;
}

// -----------------------------------------------------------------------------
// write_svg_two_series: very small SVG drawer (kept for optional local use)
// -----------------------------------------------------------------------------
void Viewer::write_svg_two_series(const std::string &fname,
                                  const std::vector<double> &x,
                                  const std::vector<double> &y1,
                                  const std::vector<double> &y2,
                                  const std::string &label1,
                                  const std::string &label2,
                                  const std::string &title)
{
    if (x.empty() || y1.size() != x.size() || y2.size() != x.size()) return;

    const int W = 800, H = 400;
    const int padL = 60, padR = 20, padT = 30, padB = 60;
    double xmin = *std::min_element(x.begin(), x.end());
    double xmax = *std::max_element(x.begin(), x.end());
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    for (size_t i=0;i<x.size();++i){
        if (std::isfinite(y1[i])) { ymin = std::min(ymin, y1[i]); ymax = std::max(ymax, y1[i]); }
        if (std::isfinite(y2[i])) { ymin = std::min(ymin, y2[i]); ymax = std::max(ymax, y2[i]); }
    }
    if (!std::isfinite(ymin) || !std::isfinite(ymax)) return;
    if (ymin == ymax) { ymin -= 1.0; ymax += 1.0; }

    std::ofstream svg(fname);
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    svg << "<svg width=\"" << W << "\" height=\"" << H << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    svg << "<rect x=\"0\" y=\"0\" width=\"" << W << "\" height=\"" << H << "\" fill=\"white\" />\n";
    svg << "<text x=\"" << (W/2) << "\" y=\"16\" font-family=\"sans-serif\" font-size=\"14\" text-anchor=\"middle\">" << title << "</text>\n";

    auto tx = [&](double xv) {
        double t = (xv - xmin) / (xmax - xmin);
        return padL + t * (W - padL - padR);
    };
    auto ty = [&](double yv) {
        double t = (yv - ymin) / (ymax - ymin);
        return H - padB - t * (H - padT - padB);
    };

    svg << "<line x1=\"" << padL << "\" y1=\"" << padT << "\" x2=\"" << padL << "\" y2=\"" << (H-padB) << "\" stroke=\"#000\" />\n";
    svg << "<line x1=\"" << padL << "\" y1=\"" << (H-padB) << "\" x2=\"" << (W-padR) << "\" y2=\"" << (H-padB) << "\" stroke=\"#000\" />\n";

    // polyline for y1 (blue)
    svg << "<polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"2\" points='";
    for (size_t i=0;i<x.size();++i) svg << tx(x[i]) << "," << ty(y1[i]) << " ";
    svg << "' />\n";
    // polyline for y2 (red)
    svg << "<polyline fill=\"none\" stroke=\"#d62728\" stroke-width=\"2\" points='";
    for (size_t i=0;i<x.size();++i) svg << tx(x[i]) << "," << ty(y2[i]) << " ";
    svg << "' />\n";
    svg << "</svg>\n";
    svg.close();
}

// -----------------------------------------------------------------------------
// run_full_evaluation_and_save: orchestrates experiments & writes CSV outputs + plots
// -----------------------------------------------------------------------------

void Viewer::run_full_evaluation_and_save(int laplace, int baseline_min_point, int steps, int sh_band, int sh_m)
{
    // local copy of face infos (must be prepared on main thread)
    std::vector<FaceInfo> local_face_infos = this->face_infos_;
    if (local_face_infos.empty()) {
        std::cerr << "[EVAL] No face_infos_ available. Call compute_face_info(...) on the main thread before running this.\n";
        return;
    }

    // prepare outputs
    std::string mesh_base = sanitize_mesh_name(mesh_filename_);
    if (mesh_base.empty()) mesh_base = "mesh_eval";
    std::string out_dir = mesh_base + "_eval_outputs";
    try { std::filesystem::create_directories(out_dir); } catch (...) {}

    auto out_path = [&](const std::string &fname)->std::string { return out_dir + "/" + fname; };

    // CSV header
    {
        std::ofstream sf(out_path("spectral_metrics.csv"));
        if (sf) {
            sf << "mesh,fraction,cond_after,lam_min_gen,lam_max_gen,kappa_gen,S_trace";
            sf << ",cg_iters,cg_time_s,poisson_avg_err,poisson_max_err";
            sf << ",avg_vertex_disp,max_vertex_disp,mean_area_rel_change,max_area_rel_change,min_triangle_angle_deg,percent_faces_below_10deg,num_flipped_faces\n";
            sf.close();
        } else {
            std::cerr << "[EVAL] ERROR: cannot create " << out_path("spectral_metrics.csv") << "\n";
        }
    }

    // sort faces by improvement (delta)
    size_t F = local_face_infos.size();
    std::vector<int> idxs(F);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), [&](int a, int b){
        return local_face_infos[a].delta > local_face_infos[b].delta;
    });

    // assemble_from_local helper
    auto assemble_from_local = [&](const std::vector<char>& mask,
                                  Eigen::SparseMatrix<double>& S_sym_out,
                                  Eigen::SparseMatrix<double>& M_out)->bool
    {
        using Triplet = Eigen::Triplet<double>;
        int nVerts = 0;
        for (const auto &fi : local_face_infos) {
            for (int vid : fi.vidx) nVerts = std::max(nVerts, vid + 1);
        }
        if (nVerts == 0) return false;

        std::vector<Triplet> triplets;
        triplets.reserve(local_face_infos.size() * 9);
        for (size_t i = 0; i < local_face_infos.size(); ++i) {
            const auto &fi = local_face_infos[i];
            int n = (int)fi.vidx.size();
            const Eigen::MatrixXd &Sbase = fi.S_baseline;
            const Eigen::MatrixXd &Strace = fi.S_trace;
            const Eigen::MatrixXd &Slocal = (mask.size() ? (mask[i] ? Strace : Sbase) : Sbase);
            for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) {
                double v = Slocal(r,c);
                if (std::abs(v) > 1e-18) triplets.emplace_back(fi.vidx[r], fi.vidx[c], v);
            }
        }

        Eigen::SparseMatrix<double> S_mixed(nVerts, nVerts);
        S_mixed.setFromTriplets(triplets.begin(), triplets.end());
        S_mixed.makeCompressed();

        // symmetric stiffness
        S_sym_out = 0.5 * (S_mixed + Eigen::SparseMatrix<double>(S_mixed.transpose()));
        S_sym_out.makeCompressed();

        // --- DEBUG + FIX: ensure stiffness has positive trace (Laplacian-like) ---
        double trS = 0.0;
        for (int k = 0; k < S_sym_out.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(S_sym_out, k); it; ++it) {
                if (it.row() == it.col()) trS += it.value();
            }
        }
        if (trS < 0.0) {
            S_sym_out = -S_sym_out;
            std::cerr << "[DBG] assemble_from_local: flipped sign of S_sym_out (trace was negative)\n";
        }

        // Mass matrix (read-only on mesh_)
        try {
            Eigen::SparseMatrix<double> Mtmp;
            setup_mass_matrices(this->mesh_, Mtmp, laplace, baseline_min_point);
            M_out = Mtmp;
            M_out.makeCompressed();
        } catch (...) {
            std::cerr << "[EVAL] Warning: setup_mass_matrices failed inside worker\n";
            return false;
        }
        return true;
    };

    // iterate fractions
    for (int s = 0; s < steps; ++s)
    {
        double frac = (steps == 1) ? 1.0 : double(s) / double(steps - 1);
        int K = std::max(0, (int)std::round(frac * double(F)));
        std::vector<char> mask(F, 0);
        for (int i = 0; i < K; ++i) mask[idxs[i]] = 1;

        Eigen::SparseMatrix<double> S_sym;
        Eigen::SparseMatrix<double> M;
        if (!assemble_from_local(mask, S_sym, M)) {
            std::cerr << "[EVAL] assemble_from_local failed at frac=" << frac << " — skipping\n";
            continue;
        }

        // dump matrices
        std::string sname = out_path(mesh_base + "_S7_frac" + std::to_string(int(100.0*frac)) + ".mtx");
        std::string mname = out_path(mesh_base + "_M7_frac" + std::to_string(int(100.0*frac)) + ".mtx");
        try {
            writeSparseMatrixMarket(S_sym, sname);
            writeSparseMatrixMarket(M, mname);
            std::cout << "[DUMP] wrote S7: " << sname << " M7: " << mname << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[EVAL][ERR] .mtx write failed at frac=" << frac << " : " << e.what() << "\n";
        } catch (...) {
            std::cerr << "[EVAL][ERR] unknown .mtx write failure at frac=" << frac << "\n";
        }

        // generalized spectral metrics
        double lam_min_gen = std::nan("");
        double lam_max_gen = std::nan("");
        double kappa_gen = std::numeric_limits<double>::infinity();
        try {
            bool ok = false;
            double lm = std::nan(""), lM = std::nan(""), kk = std::numeric_limits<double>::infinity();
            try {
                auto triple = compute_generalized_spectral_metrics(S_sym, M);
                lm = std::get<0>(triple);
                lM = std::get<1>(triple);
                kk = std::get<2>(triple);
                ok = true;
            } catch (const std::exception &e) {
                std::cerr << "[EVAL] compute_generalized_spectral_metrics exception: " << e.what() << "\n";
            } catch (...) {
                std::cerr << "[EVAL] compute_generalized_spectral_metrics unknown exception\n";
            }

            // fallback if bad lam_min
            double tol = safe_positive_tol(lM);
            if (!ok || !std::isfinite(lm) || lm <= tol) {
                double trS_local = 0.0, trM_local = 0.0;
                for (int k = 0; k < S_sym.outerSize(); ++k)
                    for (Eigen::SparseMatrix<double>::InnerIterator it(S_sym, k); it; ++it)
                        if (it.row() == it.col()) trS_local += it.value();
                for (int k = 0; k < M.outerSize(); ++k)
                    for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it)
                        if (it.row() == it.col()) trM_local += it.value();

                double diag_max_S = 0.0;
                for (int k = 0; k < S_sym.outerSize(); ++k)
                    for (Eigen::SparseMatrix<double>::InnerIterator it(S_sym, k); it; ++it)
                        if (it.row() == it.col()) diag_max_S = std::max(diag_max_S, it.value());

                if (!std::isfinite(lM)) lM = diag_max_S > 0.0 ? diag_max_S : std::nan("");
                if (trM_local > 0.0 && std::isfinite(trS_local)) {
                    lm = (trS_local / trM_local) / std::max(1.0, double(S_sym.rows()));
                } else {
                    lm = std::max(1e-16, diag_max_S * 1e-12);
                }
                if (lm <= 0.0) lm = std::abs(lm) + 1e-16;
                kk = (std::isfinite(lM) && lm > 0.0) ? std::abs(lM / lm) : std::numeric_limits<double>::infinity();
                std::cerr << "[EVAL] Fallback generalized spectral metrics used: lam_min=" << lm << " lam_max=" << lM << " kappa=" << kk << "\n";
            }

            lam_min_gen = lm;
            lam_max_gen = lM;
            kappa_gen   = kk;
        } catch (...) {
            std::cerr << "[EVAL] Unexpected error computing spectral metrics; assigning safe defaults\n";
            lam_min_gen = 1e-16;
            lam_max_gen = 1e-6;
            kappa_gen = std::abs(lam_max_gen/lam_min_gen);
        }

        // condition number of application matrix (use -S_sym if that's what your solver uses)
        double cond_after = std::numeric_limits<double>::infinity();
        try {
            Eigen::SparseMatrix<double> negS = -S_sym; negS.makeCompressed();
            cond_after = compute_condition_number(negS);
        } catch (const std::exception &e) {
            std::cerr << "[EVAL] compute_condition_number exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[EVAL] compute_condition_number unknown exception\n";
        }

        // trace of S_sym
        double traceS = 0.0;
        try {
            for (int k = 0; k < S_sym.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(S_sym, k); it; ++it)
                    if (it.row() == it.col()) traceS += it.value();
        } catch (...) {
            traceS = std::nan("");
        }

        // --- Practical metrics: CG timings, Poisson errors, geometry placeholders ---
        double cg_iters = std::nan(""), cg_time = std::nan("");
        double poisson_avg_err = std::nan(""), poisson_max_err = std::nan("");
        double avg_disp = std::nan(""), max_disp = std::nan("");
        double mean_area_rel = std::nan(""), max_area_rel = std::nan("");
        double min_tri_angle = std::nan(""), percent_below_10 = std::nan("");
        int num_flips = -1;

        // inner vertices
        std::vector<int> innerVertIdxs;
        for (auto v : mesh_.vertices()) if (!mesh_.is_boundary(v)) innerVertIdxs.push_back(v.idx());
        int nInner = (int)innerVertIdxs.size();

        if (nInner > 0) {
            // build sliced S_in_in and M_in_in
            std::vector<int> g2l(S_sym.rows(), -1);
            for (int i = 0; i < nInner; ++i) g2l[innerVertIdxs[i]] = i;
            std::vector<Eigen::Triplet<double>> Strip, Mtrip;
            for (int k = 0; k < S_sym.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(S_sym, k); it; ++it) {
                    int gi = it.row(), gj = it.col();
                    int li = g2l[gi], lj = g2l[gj];
                    if (li != -1 && lj != -1) Strip.emplace_back(li, lj, it.value());
                }
            }
            for (int k = 0; k < M.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
                    int gi = it.row(), gj = it.col();
                    int li = g2l[gi], lj = g2l[gj];
                    if (li != -1 && lj != -1) Mtrip.emplace_back(li, lj, it.value());
                }
            }

            Eigen::SparseMatrix<double> S_in_in(nInner, nInner), M_in_in(nInner, nInner);
            S_in_in.setFromTriplets(Strip.begin(), Strip.end()); S_in_in.makeCompressed();
            M_in_in.setFromTriplets(Mtrip.begin(), Mtrip.end()); M_in_in.makeCompressed();

            // quick non-finite check
            if (sparse_matrix_has_nonfinite(S_in_in) || sparse_matrix_has_nonfinite(M_in_in)) {
                std::cerr << "[EVAL] Warning: non-finite entries detected in sliced matrices; skipping CG/Poisson for frac="<<frac<<"\n";
            } else {
                // ensure S_in_in is numerically positive definite for CG: quick LLT check, otherwise regularize diagonal
                {
                    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt_check;
                    llt_check.compute(S_in_in);
                    if (llt_check.info() != Eigen::Success) {
                        // compute diag magnitude to pick eps
                        double diag_max = 0.0;
                        for (int k = 0; k < S_in_in.outerSize(); ++k)
                            for (Eigen::SparseMatrix<double>::InnerIterator it(S_in_in, k); it; ++it)
                                if (it.row()==it.col()) diag_max = std::max(diag_max, std::abs(it.value()));
                        double eps = std::max(1e-14, diag_max * 1e-10); // small regularization tuned to scale
                        std::vector<Eigen::Triplet<double>> diag_add;
                        diag_add.reserve(nInner);
                        for (int i = 0; i < nInner; ++i) diag_add.emplace_back(i, i, eps);
                        Eigen::SparseMatrix<double> S_reg(nInner, nInner);
                        // merge existing + eps diag
                        std::vector<Eigen::Triplet<double>> all = Strip;
                        all.insert(all.end(), diag_add.begin(), diag_add.end());
                        S_reg.setFromTriplets(all.begin(), all.end());
                        S_reg.makeCompressed();
                        // replace sliced matrix with regularized one
                        S_in_in.swap(S_reg);
                        std::cerr << "[EVAL] Regularized S_in_in with eps=" << eps << " to fix LLT failures (frac="<<frac<<")\n";
                    }
                }

                // CG benchmark (representative RHS)
                try {
                    Eigen::VectorXd u_true(nInner);
                    auto points = mesh_.vertex_property<Point>("v:point");
                    for (int i = 0; i < nInner; ++i) {
                        Point p = points[Vertex(innerVertIdxs[i])];
                        u_true(i) = sphericalHarmonic(p, 1, 0);
                    }
                    double coef = 1.0 * (1.0 + 1.0);
                    Eigen::VectorXd rhs = M_in_in * (coef * u_true);
                    Eigen::VectorXd x(nInner);
                    int maxcg = 20000;
                    auto [iters, tsec] = run_cg_count(S_in_in, rhs, x, 1e-10, maxcg);

                    if (iters >= maxcg) {
                        std::cerr << "[EVAL] CG hit max iterations ("<<maxcg<<") at frac="<<frac<<" — marking cg_iters=-1\n";
                        cg_iters = -1.0;
                        cg_time = -1.0;
                    } else {
                        cg_iters = double(iters);
                        cg_time = tsec;
                    }
                    if (std::isfinite(cg_time) && cg_time > 1e6) cg_time = -1.0;
                } catch (const std::exception &e) {
                    std::cerr << "[EVAL] CG exception: " << e.what() << "\n";
                } catch (...) {
                    std::cerr << "[EVAL] CG unknown exception\n";
                }

                // Poisson errors across spherical harmonics up to sh_band
                try {
                    std::vector<double> per_errs;
                    for (int l = 1; l <= sh_band; ++l) {
                        for (int m = -l; m <= l; ++m) {
                            double out_cg_time=0.0, out_iters_d=0.0; int out_iters_i=0;
                            double err = compute_poisson_L2_error(S_sym, M, l, m, out_cg_time, out_iters_d, out_iters_i);
                            if (std::isfinite(err) && err < 1e308) per_errs.push_back(err);
                        }
                    }
                    if (!per_errs.empty()) {
                        poisson_avg_err = std::accumulate(per_errs.begin(), per_errs.end(), 0.0) / double(per_errs.size());
                        poisson_max_err = *std::max_element(per_errs.begin(), per_errs.end());
                    } else {
                        poisson_avg_err = std::nan("");
                        poisson_max_err = std::nan("");
                    }
                } catch (const std::exception &e) {
                    std::cerr << "[EVAL] Poisson L2 error exception: " << e.what() << "\n";
                } catch (...) {
                    std::cerr << "[EVAL] Poisson L2 unknown exception\n";
                }
            }
        } // end if nInner > 0

        // Geometry placeholders (documented)
        double cur_min_angle = std::numeric_limits<double>::infinity();
        int faces_below_10 = 0;
        int total_faces = 0;
        for (auto f : mesh_.faces()) {
            std::vector<pmp::Point> pts;
            auto h0 = mesh_.halfedge(f);
            auto h = h0;
            do { pts.push_back(mesh_.position(mesh_.to_vertex(h))); h = mesh_.next_halfedge(h); } while (h != h0);
            if (pts.size() >= 3) {
                total_faces++;
                double local_min = std::numeric_limits<double>::infinity();
                for (size_t i=1;i+1<pts.size();++i) {
                    local_min = std::min(local_min, min_triangle_angle_deg(pts[0], pts[i], pts[i+1]));
                }
                if (std::isfinite(local_min)) {
                    cur_min_angle = std::min(cur_min_angle, local_min);
                    if (local_min < 10.0) faces_below_10++;
                }
            }
        }
        if (!std::isfinite(cur_min_angle)) cur_min_angle = 0.0;

        avg_disp = 0.0;
        max_disp = 0.0;
        mean_area_rel = 0.0;
        max_area_rel = 0.0;
        min_tri_angle = cur_min_angle;
        percent_below_10 = (total_faces>0) ? (100.0*double(faces_below_10)/double(total_faces)) : 0.0;
        num_flips = 0;

        // sanitize and write CSV line
        double lam_min_rec = lam_min_gen;
        double lam_max_rec = lam_max_gen;
        double trS_rec = traceS;
        double cond_after_rec = cond_after;

        if (std::isfinite(lam_max_rec) && lam_max_rec < 0.0 && std::abs(lam_max_rec) < 1e-12) lam_max_rec = std::abs(lam_max_rec);

        double tol_small = 1e-14;
        if (std::isfinite(lam_max_rec)) tol_small = std::max(1e-14, std::abs(lam_max_rec) * 1e-12);
        if (std::isfinite(lam_min_rec) && lam_min_rec <= tol_small) lam_min_rec = std::numeric_limits<double>::quiet_NaN();

        double kappa_rec = std::numeric_limits<double>::infinity();
        if (std::isfinite(lam_max_rec) && std::isfinite(lam_min_rec) && lam_min_rec > 0.0) {
            kappa_rec = std::abs(lam_max_rec / lam_min_rec);
        }

        std::ostringstream sline;
        sline << mesh_base << "," << std::setprecision(12) << frac << ",";
        if (std::isfinite(cond_after_rec)) sline << cond_after_rec << ","; else sline << "inf,";
        if (std::isfinite(lam_min_rec))     sline << lam_min_rec     << ","; else sline << "nan,";
        if (std::isfinite(lam_max_rec))     sline << lam_max_rec     << ","; else sline << "nan,";
        if (std::isfinite(kappa_rec))       sline << kappa_rec       << ","; else sline << "inf,";
        if (std::isfinite(trS_rec))         sline << trS_rec << ",";      else sline << "nan,";
        if (std::isfinite(cg_iters))        sline << cg_iters << ",";     else sline << "nan,";
        if (std::isfinite(cg_time))         sline << cg_time << ",";      else sline << "nan,";
        if (std::isfinite(poisson_avg_err)) sline << poisson_avg_err << ","; else sline << "nan,";
        if (std::isfinite(poisson_max_err)) sline << poisson_max_err << ","; else sline << "nan,";
        if (std::isfinite(avg_disp)) sline << avg_disp << ","; else sline << "nan,";
        if (std::isfinite(max_disp)) sline << max_disp << ","; else sline << "nan,";
        if (std::isfinite(mean_area_rel)) sline << mean_area_rel << ","; else sline << "nan,";
        if (std::isfinite(max_area_rel))  sline << max_area_rel << ",";  else sline << "nan,";
        if (std::isfinite(min_tri_angle)) sline << min_tri_angle << ","; else sline << "nan,";
        if (std::isfinite(percent_below_10)) sline << percent_below_10 << ","; else sline << "nan,";
        if (num_flips >= 0) sline << num_flips; else sline << "nan";

        append_csv_line(out_path("spectral_metrics.csv"), sline.str());
    } // end fractions loop

    std::cout << "[EVAL] Finished spectral-only evaluation. spectral_metrics.csv in: " << out_dir << std::endl;
}
// -----------------------------------------------------------------------------
// Compute and cache per-face baseline/trace local matrices and vertex indices.
// -----------------------------------------------------------------------------
void Viewer::compute_face_info(int baseline_min_point)
{
    face_infos_.clear();
    face_infos_.reserve(mesh_.n_faces());

    for (auto f : mesh_.faces())
    {
        Eigen::MatrixXd poly;
        get_polygon_from_face(mesh_, f, poly);

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

        // vertex indices in face order
        std::vector<int> vids;
        auto h0 = mesh_.halfedge(f);
        auto h = h0;
        do {
            vids.push_back(mesh_.to_vertex(h).idx());
            h = mesh_.next_halfedge(h);
        } while (h != h0);

        FaceInfo fi;
        fi.f = f;
        fi.delta = delta;
        fi.S_baseline = Si_baseline;
        fi.S_trace = Si_trace;
        fi.vidx = vids;
        fi.v_baseline = v_baseline;
        fi.v_trace    = v_trace;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs_base(Si_baseline);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs_trace(Si_trace);
        double local_min_trace = eigs_trace.eigenvalues().minCoeff();
        double local_max_trace = eigs_trace.eigenvalues().maxCoeff();
        double local_tol = std::max(1e-14, std::abs(local_max_trace) * 1e-12);
        fi.local_valid = (local_min_trace > -local_tol);

        face_infos_.push_back(fi);
    }

    if (!face_infos_.empty())
    {
        std::vector<char> mask_area(face_infos_.size(), 0); // all baseline
        std::vector<char> mask_trace(face_infos_.size(), 1); // all trace

        auto [cond_area, _unused_after] = assemble_and_compute_cond(mask_area);
        auto [_unused_base, cond_trace] = assemble_and_compute_cond(mask_trace);

        cond_area_cached_  = cond_area;
        cond_trace_cached_ = cond_trace;
    }
    else
    {
        cond_area_cached_ = cond_trace_cached_ = std::numeric_limits<double>::quiet_NaN();
    }

    update_virtual_vv_buffers();
}

// Assemble global stiffness from face_infos_ using use_trace_mask (size == face_infos_.size())
// Returns pair(cond_base, cond_after) with positive values (uses -S convention internally)
std::pair<double,double> Viewer::assemble_and_compute_cond(const std::vector<char>& use_trace_mask)
{
    using Triplet = Eigen::Triplet<double>;
    int nVerts = mesh_.n_vertices();

    std::vector<Triplet> mixed_triplets;
    mixed_triplets.reserve(face_infos_.size()*9);
    std::vector<Triplet> base_triplets;
    base_triplets.reserve(face_infos_.size()*9);

    for (size_t i = 0; i < face_infos_.size(); ++i)
    {
        const auto &fi = face_infos_[i];
        int n = (int)fi.vidx.size();
        const Eigen::MatrixXd &Sbase = fi.S_baseline;
        const Eigen::MatrixXd &Strace = fi.S_trace;
        const Eigen::MatrixXd &Slocal = (use_trace_mask.size() ? (use_trace_mask[i] ? Strace : Sbase) : Sbase);
        for (int r=0;r<n;++r) for (int c=0;c<n;++c)
        {
            double vM = Slocal(r,c);
            if (std::abs(vM) > 1e-16) mixed_triplets.emplace_back(fi.vidx[r], fi.vidx[c], vM);
            double vB = Sbase(r,c);
            if (std::abs(vB) > 1e-16) base_triplets.emplace_back(fi.vidx[r], fi.vidx[c], vB);
        }
    }

    Eigen::SparseMatrix<double> S_mixed(nVerts, nVerts), S_base(nVerts, nVerts);
    S_mixed.setFromTriplets(mixed_triplets.begin(), mixed_triplets.end());
    S_mixed.makeCompressed();
    S_base.setFromTriplets(base_triplets.begin(), base_triplets.end());
    S_base.makeCompressed();

    Eigen::SparseMatrix<double> S_mixed_sym = 0.5 * (S_mixed + Eigen::SparseMatrix<double>(S_mixed.transpose()));
    S_mixed_sym.makeCompressed();
    Eigen::SparseMatrix<double> S_base_sym = 0.5 * (S_base + Eigen::SparseMatrix<double>(S_base.transpose()));
    S_base_sym.makeCompressed();

    Eigen::SparseMatrix<double> neg_mixed = -S_mixed_sym; neg_mixed.makeCompressed();
    Eigen::SparseMatrix<double> neg_base  = -S_base_sym;  neg_base.makeCompressed();

    double cond_after = compute_condition_number(neg_mixed);
    double cond_base  = compute_condition_number(neg_base);

    return {cond_base, cond_after};
}

// steps: number of sample points (e.g., 20), baseline_min_point same as used in UI
void Viewer::run_selective_sweep(int steps, int baseline_min_point)
{
    std::vector<double> local_fractions;
    std::vector<double> local_cond_base;
    std::vector<double> local_cond_after;

    if (face_infos_.empty()) compute_face_info(baseline_min_point);

    size_t F = face_infos_.size();
    std::vector<int> idxs(F);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), [&](int a, int b){
        return face_infos_[a].delta > face_infos_[b].delta;
    });

    for (int s = 0; s < steps; ++s)
    {
        double frac = (steps==1)?1.0: double(s) / double(steps-1); // 0..1 inclusive
        int K = std::max(0, (int)std::round(frac * double(F)));
        std::vector<char> mask(F, 0);
        for (int i=0;i<K;++i) mask[idxs[i]] = 1;

        auto [cond_base, cond_after] = assemble_and_compute_cond(mask);
        local_fractions.push_back(frac);
        local_cond_base.push_back(cond_base);
        local_cond_after.push_back(cond_after);
    }

    {
        std::lock_guard<std::mutex> lock(dash_mutex_);
        dash_fractions_ = std::move(local_fractions);
        dash_cond_base_ = std::move(local_cond_base);
        dash_cond_after_ = std::move(local_cond_after);
    }
}

// -----------------------------------------------------------------------------
// color_code_condition_numbers: use dense local eigen solver to color faces
// -----------------------------------------------------------------------------
void Viewer::color_code_condition_numbers(int laplace, int min_point)
{
    auto face_color = mesh_.face_property<Color>("f:color");
    auto face_cond = mesh_.face_property<double>("f:condition");

    Eigen::Vector3d values;
    double cond = condition_number(mesh_, laplace, min_point, values, false);
    std::cout << "Condition Number: " << cond << std::endl;

    Eigen::MatrixXd Si;
    Eigen::VectorXd w;
    Eigen::Vector3d p;
    Eigen::MatrixXd poly;
    for (Face f : mesh_.faces())
    {
        get_polygon_from_face(mesh_, f, poly);

        if (min_point == Centroid_)
        {
            int val = (int)poly.rows();
            w = Eigen::VectorXd::Ones(val);
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
        Eigen::Vector3d min = poly.transpose() * w;

        localCotanMatrix(poly, min, w, Si);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(Si);
        Eigen::VectorXd ev = eigs.eigenvalues();
        if (ev.size() >= 2 && ev[1] != 0.0) {
            face_cond[f] = ev[ev.size() - 1] / ev[1];
        } else {
            face_cond[f] = std::numeric_limits<double>::infinity();
        }
    }

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

// -----------------------------------------------------------------------------
// draw: render mesh + virtual vertices
// -----------------------------------------------------------------------------
void Viewer::draw(const std::string& draw_mode)
{
    renderer_.draw(projection_matrix_, modelview_matrix_, draw_mode);
    render_virtual_vertices();

    if (draw_mode == "Texture" && show_uv_layout_)
    {
        glClear(GL_DEPTH_BUFFER_BIT);
        GLint size = std::min(width(), height()) / 4;
        glViewport(width() - size - 1, height() - size - 1, size, size);
        mat4 P = ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        mat4 M = mat4::identity();
        renderer_.draw(P, M, "Texture Layout");
        glViewport(0, 0, width(), height());
    }
}

// -----------------------------------------------------------------------------
// render_virtual_vertices: VBO-based rendering
// -----------------------------------------------------------------------------
void Viewer::render_virtual_vertices()
{
    if (!show_virtual_vertices_ || !vv_gl_initialized_ ) return;
    if (vv_point_count_ == 0 && vv_line_vertex_count_ == 0) return;

    float mvp[16];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float val = 0.0f;
            for (int k = 0; k < 4; ++k) {
                val += projection_matrix_(r, k) * modelview_matrix_(k, c);
            }
            mvp[c * 4 + r] = val; // column-major
        }
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(vv_prog_);

    GLint loc_mvp = glGetUniformLocation(vv_prog_, "uMVP");
    GLint loc_ps = glGetUniformLocation(vv_prog_, "uPointSize");
    if (loc_mvp >= 0) glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
    if (loc_ps >= 0) glUniform1f(loc_ps, virtual_point_size_);

    glBindVertexArray(vv_vao_);

    if (vv_point_count_ > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, vv_points_vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)(sizeof(float)*3));

        glDrawArrays(GL_POINTS, 0, vv_point_count_);
    }

    if (vv_line_vertex_count_ > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, vv_lines_vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)(sizeof(float)*3));

        glDrawArrays(GL_LINES, 0, vv_line_vertex_count_);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
}

// -----------------------------------------------------------------------------
// update_virtual_vv_buffers: upload face_infos_ -> GL buffers
// -----------------------------------------------------------------------------
void Viewer::update_virtual_vv_buffers()
{
    if (!vv_gl_initialized_) init_virtual_vv_gl();
    std::vector<float> point_data; // pos.xyz, col.rgb
    std::vector<float> line_data;  // pairs of vertices

    auto vpos = mesh_.vertex_property<Point>("v:point");

    for (size_t i = 0; i < face_infos_.size(); ++i) {
        const auto &fi = face_infos_[i];
        auto showprop = mesh_.face_property<char>("f:show_vv");
        if (show_only_selected_face_vv_) {
            if (!showprop || !showprop[fi.f]) continue;
        }

        if (virtual_vertex_mode_ == 0 || virtual_vertex_mode_ == 2) {
            Eigen::Vector3d p = fi.v_baseline;
            point_data.push_back((float)p[0]); point_data.push_back((float)p[1]); point_data.push_back((float)p[2]);
            point_data.push_back(1.0f); point_data.push_back(0.2f); point_data.push_back(0.2f);
            for (int vid : fi.vidx) {
                Point vp = vpos[Vertex(vid)];
                line_data.push_back((float)p[0]); line_data.push_back((float)p[1]); line_data.push_back((float)p[2]);
                line_data.push_back(1.0f); line_data.push_back(0.2f); line_data.push_back(0.2f);
                line_data.push_back((float)vp[0]); line_data.push_back((float)vp[1]); line_data.push_back((float)vp[2]);
                line_data.push_back(0.2f); line_data.push_back(0.2f); line_data.push_back(0.2f);
            }
        }

        if (virtual_vertex_mode_ == 1 || virtual_vertex_mode_ == 2) {
            Eigen::Vector3d p = fi.v_trace;
            point_data.push_back((float)p[0]); point_data.push_back((float)p[1]); point_data.push_back((float)p[2]);
            point_data.push_back(0.2f); point_data.push_back(0.9f); point_data.push_back(0.2f);
            for (int vid : fi.vidx) {
                Point vp = vpos[Vertex(vid)];
                line_data.push_back((float)p[0]); line_data.push_back((float)p[1]); line_data.push_back((float)p[2]);
                line_data.push_back(0.2f); line_data.push_back(0.9f); line_data.push_back(0.2f);
                line_data.push_back((float)vp[0]); line_data.push_back((float)vp[1]); line_data.push_back((float)vp[2]);
                line_data.push_back(0.2f); line_data.push_back(0.2f); line_data.push_back(0.2f);
            }
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vv_points_vbo_);
    if (!point_data.empty())
        glBufferData(GL_ARRAY_BUFFER, point_data.size() * sizeof(float), point_data.data(), GL_DYNAMIC_DRAW);
    else
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, vv_lines_vbo_);
    if (!line_data.empty())
        glBufferData(GL_ARRAY_BUFFER, line_data.size() * sizeof(float), line_data.data(), GL_DYNAMIC_DRAW);
    else
        glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    vv_point_count_ = (int)(point_data.size() / 6);
    vv_line_vertex_count_ = (int)(line_data.size() / 6);
}

// primary experiment runner (call from UI or main to run experiments for currently loaded mesh)
void Viewer::run_experiments(const std::string &output_prefix,
                            const std::vector<std::string> &sto_strategies,
                            const std::vector<double> &fractions,
                            const std::vector<int> &taus,
                            int numIters,
                            int use_laplace,
                            int use_baseline_min_point)
{
    using namespace std::chrono;

    if (mesh_.n_vertices() == 0) {
        std::cerr << "[run_experiments] no mesh loaded\n";
        return;
    }

    // prepare global CSV (extended header to include CG/Poisson/geometry metrics)
    std::string global_csv = output_prefix + "_global_results.csv";
    {
        std::ofstream gcsv(global_csv);
        gcsv << "model_name,resolution_tag,method,sto_strategy,fraction,quadricsTau,num_iters,total_runtime_sec,avg_runtime_per_iter_sec,initial_condition_number,final_condition_number";
        gcsv << ",cg_iters,cg_time_s,poisson_avg_err,poisson_max_err";
        gcsv << ",avg_vertex_disp,max_vertex_disp,mean_area_rel_change,max_area_rel_change,min_triangle_angle_deg,percent_faces_below_10deg,num_flipped_faces\n";
    }

    // capture original mesh positions
    std::vector<pmp::Point> pos_before(mesh_.n_vertices());
    for (auto v : mesh_.vertices()) pos_before[v.idx()] = mesh_.position(v);

    // compute initial condition number baseline (all baseline)
    compute_face_info(use_baseline_min_point); // ensure face_infos_ computed
    std::vector<char> mask_none(face_infos_.size(), 0);
    auto [cond_base_init, cond_after_init] = assemble_and_compute_cond(mask_none);
    double initial_cnum = cond_base_init;

    // helper: build masks for selection strategies given fraction
    auto build_mask_for_strategy = [&](const std::string &strategy, double frac)->std::vector<char> {
        size_t F = face_infos_.size();
        std::vector<int> idxs(F);
        std::iota(idxs.begin(), idxs.end(), 0);
        if (strategy == "trace_top") {
            std::stable_sort(idxs.begin(), idxs.end(), [&](int a,int b){ return face_infos_[a].delta > face_infos_[b].delta; });
        } else if (strategy == "cond_top") {
            // approximate by baseline local trace magnitude
            std::stable_sort(idxs.begin(), idxs.end(), [&](int a,int b){
                double ta = face_infos_[a].S_baseline.trace();
                double tb = face_infos_[b].S_baseline.trace();
                return ta > tb;
            });
        } else if (strategy == "curvature_top") {
            // approximate curvature/quality by (1/min angle) using local dense Si baseline eigenvalues
            std::stable_sort(idxs.begin(), idxs.end(), [&](int a,int b){
                double ma = face_infos_[a].delta;
                double mb = face_infos_[b].delta;
                return ma > mb;
            });
        } else { // random
            std::mt19937 rng(123456);
            std::shuffle(idxs.begin(), idxs.end(), rng);
        }
        int K = std::max(0, (int)std::round(frac * double(F)));
        std::vector<char> mask(F, 0);
        for (int i = 0; i < K; ++i) mask[idxs[i]] = 1;
        return mask;
    };

    // default evaluation spherical harmonic band for Poisson tests (used in run_single)
    const int eval_sh_band = 3;

    // Methods: Baseline, FTM, STO, AreaBaseline (optional)
    auto run_single = [&](const std::string &method,
                          const std::string &strategy,
                          double fraction,
                          int tau) {
        // restore mesh positions
        for (auto v : mesh_.vertices()) mesh_.position(v) = pos_before[v.idx()];
        update_mesh();

        // store pre-positions for later displacement calculation
        std::vector<pmp::Point> before_pos = pos_before;

        // prepare faceVirtuals property (if present) and set it according to strategy/method
        auto faceVirtuals = mesh_.face_property<Eigen::Vector3d>("f:Virtuals");
        if (!faceVirtuals) faceVirtuals = mesh_.add_face_property<Eigen::Vector3d>("f:Virtuals");

        // compute face infos if not present
        if (face_infos_.empty()) compute_face_info(use_baseline_min_point);

        // default mask (all baseline)
        std::vector<char> mask(face_infos_.size(), 0);
        if (method == "Baseline") {
            // set virtuals to baseline for all faces
            for (size_t i=0;i<face_infos_.size();++i) {
                faceVirtuals[face_infos_[i].f] = face_infos_[i].v_baseline;
            }
        }
        else if (method == "FTM") {
            // set virtuals to trace for all faces
            for (size_t i=0;i<face_infos_.size();++i) {
                faceVirtuals[face_infos_[i].f] = face_infos_[i].v_trace;
            }
        }
        else if (method == "STO") {
            mask = build_mask_for_strategy(strategy, fraction);
            for (size_t i=0;i<face_infos_.size();++i) {
                faceVirtuals[face_infos_[i].f] = mask[i] ? face_infos_[i].v_trace : face_infos_[i].v_baseline;
            }
        } else if (method == "AreaBaseline") {
            // explicitly set area-minimizer virtuals
            for (size_t i=0;i<face_infos_.size();++i) {
                faceVirtuals[face_infos_[i].f] = face_infos_[i].v_baseline;
            }
        }

        // configure smoothing
        SmoothingConfigs oConf_local = SmoothingConfigs( numIters, false, false, false, false );
        oConf_local.updateQuadrics = true; // we want quadrics updated optionally inside optimize
        oConf_local.fixBoundary = true;
        oConf_local.numIters = numIters;

        // create PolySmoothing and run optimize. It will read f:Virtuals
        PolySmoothing polyS(mesh_, oConf_local);

        // time the run
        auto t0 = high_resolution_clock::now();
        try {
            // call optimize: note optimize uses quadricsTau param currently in your code path
            polyS.optimize(tau);
        } catch (const std::exception &e) {
            std::cerr << "[run_experiments] optimize exception: " << e.what() << std::endl;
        }
        auto t1 = high_resolution_clock::now();
        double secs = duration<double>(t1 - t0).count();

        // compute post-run metrics
        // final condition number: assemble S_mixed and use compute_condition_number
        std::vector<char> use_mask = mask; // already set for STO; for FTM/Baseline all 1/0 logic handled by f:Virtuals previously
        // But safer: rebuild mask by comparing f:Virtuals to fi.v_trace / v_baseline
        for (size_t i=0;i<face_infos_.size();++i) {
            auto fv = faceVirtuals[face_infos_[i].f];
            if ((fv - face_infos_[i].v_trace).norm() < 1e-12) use_mask[i] = 1;
            else use_mask[i] = 0;
        }
        auto [condB, condA] = assemble_cond_from_faceinfos(face_infos_, use_mask);

        // --- compute post-smoothing S_mixed_sym and mass M for CG + Poisson metrics
        Eigen::SparseMatrix<double> S_mixed_sym, M_mass;
        assemble_S_mixed_and_mass(use_mask, S_mixed_sym, M_mass, /*laplace=*/use_laplace, /*baseline_min_point=*/use_baseline_min_point);

        // compute CG & Poisson metrics (similar approach as in run_full_evaluation_and_save but on current mesh state)
        double cg_iters_post = std::nan(""), cg_time_post = std::nan("");
        double poisson_avg_err_post = std::nan(""), poisson_max_err_post = std::nan("");

        // build inner mapping
        std::vector<int> innerVertIdxs;
        for (auto v : mesh_.vertices()) if (!mesh_.is_boundary(v)) innerVertIdxs.push_back(v.idx());
        int nInner = (int)innerVertIdxs.size();

        if (nInner > 0) {
            std::vector<int> g2l(S_mixed_sym.rows(), -1);
            for (int i = 0; i < nInner; ++i) g2l[innerVertIdxs[i]] = i;
            std::vector<Eigen::Triplet<double>> Strip, Mtrip;
            for (int k=0;k < S_mixed_sym.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(S_mixed_sym,k); it; ++it) {
                    int gi = it.row(), gj = it.col();
                    int li = g2l[gi], lj = g2l[gj];
                    if (li!=-1 && lj!=-1) Strip.emplace_back(li, lj, it.value());
                }
            for (int k=0;k < M_mass.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(M_mass,k); it; ++it) {
                    int gi = it.row(), gj = it.col();
                    int li = g2l[gi], lj = g2l[gj];
                    if (li!=-1 && lj!=-1) Mtrip.emplace_back(li, lj, it.value());
                }
            Eigen::SparseMatrix<double> S_in_in(nInner, nInner), M_in_in(nInner, nInner);
            S_in_in.setFromTriplets(Strip.begin(), Strip.end()); S_in_in.makeCompressed();
            M_in_in.setFromTriplets(Mtrip.begin(), Mtrip.end()); M_in_in.makeCompressed();

            // CG run: representative RHS (spherical harmonic l=1,m=0)
            try {
                Eigen::VectorXd u_true(nInner);
                auto points = mesh_.vertex_property<Point>("v:point");
                for (int i = 0; i < nInner; ++i) {
                    Point p = points[Vertex(innerVertIdxs[i])];
                    u_true(i) = sphericalHarmonic(p, 1, 0);
                }
                double coef = 1.0 * (1.0 + 1.0);
                Eigen::VectorXd rhs = M_in_in * (coef * u_true);
                Eigen::VectorXd x(nInner);
                auto [iters, time_s] = run_cg_count(S_in_in, rhs, x, 1e-10, 20000);
                cg_iters_post = double(iters);
                cg_time_post = time_s;
            } catch (...) {
            }

            // Poisson L2 error across eval_sh_band
            try {
                std::vector<double> per_errs;
                for (int l=1; l<=eval_sh_band; ++l) {
                    for (int m=-l; m<=l; ++m) {
                        double out_cg_time=0.0, out_iters_d=0.0; int out_iters_i=0;
                        double err = compute_poisson_L2_error(S_mixed_sym, M_mass, l, m, out_cg_time, out_iters_d, out_iters_i);
                        if (std::isfinite(err)) per_errs.push_back(err);
                    }
                }
                if (!per_errs.empty()) {
                    poisson_avg_err_post = std::accumulate(per_errs.begin(), per_errs.end(), 0.0) / double(per_errs.size());
                    poisson_max_err_post = *std::max_element(per_errs.begin(), per_errs.end());
                }
            } catch (...) {
            }
        }

        // compute geometry stats comparing before_pos -> current mesh
        double avg_disp=0, max_disp=0, mean_area_rel=0, max_area_rel=0;
        double min_tri_ang=0, percent_below_10=0;
        int num_flips = 0;
        compute_geometry_stats(mesh_, before_pos, avg_disp, max_disp, mean_area_rel, max_area_rel, min_tri_ang, percent_below_10, num_flips);

        // write to global CSV (extended)
        {
            std::ofstream g(global_csv, std::ios::app);
            g << sanitize_mesh_name(mesh_filename_) << ",";
            g << "med" << ","; // resolution_tag placeholder
            g << method << ",";
            g << ((strategy.empty()) ? "none" : strategy) << ",";
            g << fraction << ",";
            g << tau << ",";
            g << numIters << ",";
            g << secs << ",";
            g << (secs / std::max(1, numIters)) << ",";
            g << initial_cnum << ",";
            g << condA << ",";
            // cg & poisson
            if (std::isfinite(cg_iters_post)) g << cg_iters_post << ","; else g << "nan,";
            if (std::isfinite(cg_time_post))  g << cg_time_post << ",";  else g << "nan,";
            if (std::isfinite(poisson_avg_err_post)) g << poisson_avg_err_post << ","; else g << "nan,";
            if (std::isfinite(poisson_max_err_post)) g << poisson_max_err_post << ","; else g << "nan,";
            // geometry
            g << avg_disp << ",";
            g << max_disp << ",";
            g << mean_area_rel << ",";
            g << max_area_rel << ",";
            g << min_tri_ang << ",";
            g << percent_below_10 << ",";
            g << num_flips << "\n";
        }

        // per-run CSV (simple summary) - add some new fields
        {
            std::ostringstream runname;
            runname << output_prefix << "_" << method;
            if (!strategy.empty()) runname << "_" << strategy;
            runname << "_tau" << tau << "_f" << int(100.0*fraction) << ".csv";
            std::ofstream rs(runname.str());
            rs << "field,value\n";
            rs << "total_runtime_s," << secs << "\n";
            rs << "final_condition_number," << condA << "\n";
            rs << "cg_iters," << (std::isfinite(cg_iters_post) ? std::to_string(cg_iters_post) : std::string("nan")) << "\n";
            rs << "cg_time_s," << (std::isfinite(cg_time_post) ? std::to_string(cg_time_post) : std::string("nan")) << "\n";
            rs << "poisson_avg_err," << (std::isfinite(poisson_avg_err_post) ? std::to_string(poisson_avg_err_post) : std::string("nan")) << "\n";
            rs << "avg_vertex_disp," << avg_disp << "\n";
            rs << "max_vertex_disp," << max_disp << "\n";
        }

        std::cout << "[run_experiments] done " << method << " " << strategy << " f=" << fraction << " tau=" << tau << " time=" << secs << "s cond=" << condA << "\n";
    }; // end run_single lambda

    // Now run experiments:
    // Baseline (no smoothing)
    for (int tau : taus) {
        run_single("Baseline", "", 0.0, tau);
    }

    // Full trace minimization (FTM)
    for (int tau : taus) {
        run_single("FTM", "", 1.0, tau);
    }

    // STO for each strategy & fraction
    for (const auto &strategy : sto_strategies) {
        for (double f : fractions) {
            for (int tau : taus) {
                run_single("STO", strategy, f, tau);
            }
        }
    }

    // Optional: area-minimizer comparators
    for (int tau : taus) run_single("AreaBaseline", "", 0.0, tau);

    std::cout << "[run_experiments] All experiments finished. Global CSV: " << global_csv << std::endl;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Viewer window("Polygon Laplace Demo", 800, 600);
    if (argc == 2)
        window.load_mesh(argv[1]);
    return window.run();
}