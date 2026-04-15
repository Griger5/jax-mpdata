module mpdata_c_wrapper
  use iso_c_binding
  implicit none

contains

subroutine f_mpdata_2d(u1, u2, x, h, nx, nz, iflg, liner, v1,v2,f1,f2,cp,cn,mx,mn) bind(C, name="f_mpdata_2d")
  use iso_c_binding
  implicit none

  integer(c_int), value :: nx, nz, iflg, liner

  real(c_float) :: u1(*)
  real(c_float) :: u2(*)
  real(c_float) :: x(*)
  real(c_float) :: h(*)
  real(c_float) :: v1(*)
  real(c_float) :: v2(*)
  real(c_float) :: f1(*)
  real(c_float) :: f2(*)
  real(c_float) :: cp(*)
  real(c_float) :: cn(*)
  real(c_float) :: mx(*)
  real(c_float) :: mn(*)

  call mpdat_2d(u1, u2, x, h, nx, nz, iflg, liner, v1, v2, f1, f2, cp, cn, mx, mn)

end subroutine

end module