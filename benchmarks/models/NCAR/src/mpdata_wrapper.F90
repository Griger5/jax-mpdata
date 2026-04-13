module mpdata_c_wrapper
  use iso_c_binding
  implicit none

contains

subroutine f_mpdata_2d(u1, u2, x, h, nx, nz, iflg, liner) bind(C, name="f_mpdata_2d")
  use iso_c_binding
  implicit none

  integer(c_int), value :: nx, nz, iflg, liner

  real(c_float) :: u1(*)
  real(c_float) :: u2(*)
  real(c_float) :: x(*)
  real(c_float) :: h(*)

  call mpdat_2d(u1, u2, x, h, nx, nz, iflg, liner)

end subroutine

end module