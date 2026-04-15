#include <valarray>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

template <typename T = float>
using Array2D = nb::ndarray<T, nb::ndim<2>, nb::f_contig>;

extern "C" {
    void f_mpdata_2d(float *u1, float *u2, float *x, float *h, int nx, int nz, int iflg, int liner, float *v1, float *v2, float *f1, float *f2, float *cp, float *cn, float *mx, float *mn);
}

void mpdata_2d(Array2D<> u1, Array2D<> u2, Array2D<> x, Array2D<> h, int iflg, int liner, int nt) {
    size_t nx = x.shape(0);
    size_t nz = x.shape(1);  
    size_t n1 = nx + 1;
    size_t n2 = nz + 1;
    size_t n1m = nx;
    size_t n2m = nz;
    size_t nxz = nx * nz;

    std::valarray<float> v1(n1 * n2m);
    std::valarray<float> v2(n1m * n2);
    std::valarray<float> f1(n1 * n2);
    std::valarray<float> f2(n1m * n2);
    std::valarray<float> cp(n1m * n2m);
    std::valarray<float> cn(n1m * n2m);
    std::valarray<float> mx(n1m * n2m);
    std::valarray<float> mn(n1m * n2m);

    for (int i = 0; i < nt; i++) {
        f_mpdata_2d(u1.data(), u2.data(), x.data(), h.data(), x.shape(0), x.shape(1), iflg, liner, &v1[0], &v2[0], &f1[0], &f2[0], &cp[0], &cn[0], &mx[0], &mn[0]);
    }
}

NB_MODULE(ncar_mpdata, m) {
    m.def("mpdata_2d", &mpdata_2d, "2D MPDATA Solver");
}