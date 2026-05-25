#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <libmpdata++/solvers/mpdata.hpp>
#include <libmpdata++/concurr/serial.hpp>
#include <libmpdata++/concurr/threads.hpp>

namespace nb = nanobind;

template <typename T = float>
using Array2D = nb::ndarray<T, nb::ndim<2>, nb::c_contig>;

using namespace libmpdataxx;

Array2D<> mpdata_2d(Array2D<> advectee_np, Array2D<> u_np, Array2D<> v_np, double dt, int nt, int n_iters) {
    std::cerr.setstate(std::ios_base::failbit);
    enum {x, y};

    struct ct_params_t : ct_params_default_t {
        using real_t = float;
        enum { n_dims = 2 };
        enum { n_eqns = 1 };
        enum { opts = opts::dfl };
    };

    using slv_t = solvers::mpdata<ct_params_t>;
    typename slv_t::rt_params_t p;

    auto nx = advectee_np.shape(0);
    auto ny = advectee_np.shape(1);

    p.n_iters = n_iters;
    p.grid_size = {nx, ny};
    p.di = nx;
    p.dj = ny;
    p.dt = dt;

    concurr::threads<slv_t, bcond::cyclic, bcond::cyclic, bcond::cyclic, bcond::cyclic> slv{p};

    blitz::GeneralArrayStorage<2> cStorage;
    cStorage.ordering() = blitz::secondDim, blitz::firstDim;
    cStorage.base() = 0, 0;
    cStorage.ascendingFlag() = true, true;

    blitz::Range i{0, nx - 1};
    blitz::Range j{0, ny - 1};

    blitz::Array<ct_params_t::real_t, 2> adv(
        advectee_np.data(), blitz::shape(nx, ny), blitz::neverDeleteData, cStorage);
    blitz::Array<ct_params_t::real_t, 2> u(
        u_np.data(), blitz::shape(nx + 1, ny), blitz::neverDeleteData, cStorage);
    blitz::Array<ct_params_t::real_t, 2> v(
        v_np.data(), blitz::shape(nx, ny + 1), blitz::neverDeleteData, cStorage);

    slv.advectee()(i,j) = adv;
    slv.advector(x)(i,j) = u;
    slv.advector(y)(i,j) = v;

    slv.advance(nt);

    adv = slv.advectee()(i,j);
    
    return advectee_np;
}

NB_MODULE(libmpdataxx, m) {
    m.def("mpdata_2d", &mpdata_2d, "2D MPDATA Solver");
}