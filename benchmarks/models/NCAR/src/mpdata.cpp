#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

template <typename T = float>
using Array2D = nb::ndarray<T, nb::ndim<2>, nb::f_contig>;

extern "C" {
    void f_mpdata_2d(float *u1, float *u2, float *x, float *h, int nx, int nz, int iflg, int liner);
}

void mpdata_2d(Array2D<> u1, Array2D<> u2, Array2D<> x, Array2D<> h, int iflg, int liner) {
    f_mpdata_2d(u1.data(), u2.data(), x.data(), h.data(), x.shape(0), x.shape(1), iflg, liner);
}

NB_MODULE(ncar_mpdata, m) {
    m.def("mpdata_2d", &mpdata_2d, "2D MPDATA Solver");
}