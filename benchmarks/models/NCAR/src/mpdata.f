c     based on https://github.com/igfuw/bE_SDs/blob/master/babyEULAG.SDs.for
      subroutine mpdat_2d(u1,u2,x,h,nx,nz,iflg,liner,v1,v2,f1,f2,cp,cn
     1,mx,mn)
      integer nx, nz
      integer nxz, n1, n2, n1m, n2m
      integer i,j,k,im,ip,jm,jp,i0

      real, intent(in)    :: u1(nx+1, nz)
      real, intent(in)    :: u2(nx, nz+1)
      real, intent(inout) :: x(nx, nz)
      real, intent(in)    :: h(nx, nz)

      real, intent(inout)  :: v1(nx + 1, nz)
      real, intent(inout)  :: v2(nx, nz + 1)
      real, intent(inout)  :: f1(nx + 1, nz)
      real, intent(inout)  :: f2(nx, nz + 1)
      real, intent(inout)  :: cp(nx, nz)
      real, intent(inout)  :: cn(nx, nz)
      real, intent(inout)  :: mx(nx, nz)
      real, intent(inout)  :: mn(nx, nz)

      real, parameter :: ep = 1.e-12

      parameter(iord0=1,isor=0,nonos=0,idiv=0)
c
      n1 = nx + 1
      n2 = nz + 1
      n1m = nx
      n2m = nz
      nxz = nx * nz

      iord=iord0
      if(isor.eq.3) iord=max0(iord,3)
      if(liner.eq.1) iord=1

      do j=1,n2-1
        do i=1,n1
          v1(i,j) = u1(i,j)
        end do 
      end do 
      do i=1,n1-1
        do j=1,n2
          v2(i,j) = u2(i,j)
        end do 
      enddo

      if(nonos.eq.1) then
      do j=1,n2m
      jm=max0(j-1,1  )
      jp=min0(j+1,n2m)
      do i=1,n1m
      im=(i-1+(n1-i)/n1m*(n1-2))
      ip=(i+1    -i /n1m*(n1-2))
      mx(i,j)=amax1(x(im,j),x(i,j),x(ip,j),x(i,jm),x(i,jp))
      mn(i,j)=amin1(x(im,j),x(i,j),x(ip,j),x(i,jm),x(i,jp))
      end do
      end do
      endif
 
                         do 3 k=1,iord
 
      do 331 j=1,n2-1
      do 331 i=2,n1-1
  331 f1(i,j)=donor(x(i-1,j),x(i,j),v1(i,j))
      do j=1,n2-1
      f1(1 ,j)=f1(n1-1,j)
      f1(n1,j)=f1(2,j)
      enddo
      do 332 j=2,n2-1
      do 332 i=1,n1-1
  332 f2(i,j)=donor(x(i,j-1),x(i,j),v2(i,j))
      if (iflg.eq.6) then
        do i=1,n1m
          f2(i, 1)=donor(x(i,  1),x(i,  1),v2(i, 1))
          f2(i,n2)=donor(x(i,n2m),x(i,n2m),v2(i,n2))
        end do
      else
        do i=1,n1m
          ! f2(i, 1)=-f2(i,  2)
          ! f2(i,n2)=-f2(i,n2m)
          f2(i,1)  = f2(i,n2m)
          f2(i,n2) = f2(i,2)
        end do
      end if
  
      do 333 j=1,n2-1
      do 333 i=1,n1-1
  333 x(i,j)=x(i,j)-(f1(i+1,j)-f1(i,j)+f2(i,j+1)-f2(i,j))/h(i,j)
 
      if(k.eq.iord) go to 6

      do 49 j=1,n2-1
      do 49 i=1,n1
      f1(i,j)=v1(i,j)
   49 v1(i,j)=0.
      do 50 j=1,n2
      do 50 i=1,n1-1
      f2(i,j)=v2(i,j)
   50 v2(i,j)=0.
      do 51 j=2,n2-2
      do 51 i=2,n1-1
   51 v1(i,j)=vdyf(x(i-1,j),x(i,j),f1(i,j),.5*(h(i-1,j)+h(i,j)))
     *       +vcorr(f1(i,j), f2(i-1,j)+f2(i-1,j+1)+f2(i,j+1)+f2(i,j),
     *   abs(x(i-1,j+1))+abs(x(i,j+1))-abs(x(i-1,j-1))-abs(x(i,j-1)),
     *   abs(x(i-1,j+1))+abs(x(i,j+1))+abs(x(i-1,j-1))+abs(x(i,j-1))+ep,
     *                 .5*(h(i-1,j)+h(i,j)))
      if(idiv.eq.1) then
      do 511 j=2,n2-2
      do 511 i=2,n1-1
  511 v1(i,j)=v1(i,j)
     *    -vdiv1(f1(i-1,j),f1(i,j),f1(i+1,j),.5*(h(i-1,j)+h(i,j)))
     *    -vdiv2(f1(i,j),f2(i-1,j+1),f2(i,j+1),f2(i-1,j),f2(i,j),
     *                 .5*(h(i-1,j)+h(i,j)))
      endif
      do 52 j=2,n2-1
      do 52 i=2,n1-2
   52 v2(i,j)=vdyf(x(i,j-1),x(i,j),f2(i,j),.5*(h(i,j-1)+h(i,j)))
     *       +vcorr(f2(i,j), f1(i,j-1)+f1(i,j)+f1(i+1,j)+f1(i+1,j-1),
     *   abs(x(i+1,j-1))+abs(x(i+1,j))-abs(x(i-1,j-1))-abs(x(i-1,j)),
     *   abs(x(i+1,j-1))+abs(x(i+1,j))+abs(x(i-1,j-1))+abs(x(i-1,j))+ep,
     *                 .5*(h(i,j-1)+h(i,j)))
      i0=n1-2
      do j=2,n2-1
      v2(1,j)=vdyf(x(1,j-1),x(1,j),f2(1,j),.5*(h(1,j-1)+h(1,j)))
     *       +vcorr(f2(1,j), f1(1,j-1)+f1(1,j)+f1(2,j)+f1(2,j-1),
     *   abs(x(2,j-1))+abs(x(2,j))-abs(x(i0,j-1))-abs(x(i0,j)),
     *   abs(x(2,j-1))+abs(x(2,j))+abs(x(i0,j-1))+abs(x(i0,j))+ep,
     *                 .5*(h(1,j-1)+h(1,j)))
      v2(n1-1,j)=v2(1,j)
      enddo

      if(idiv.eq.1) then
      do 521 j=2,n2-1
      do 521 i=1,n1-1
  521 v2(i,j)=v2(i,j)
     *    -vdiv1(f2(i,j-1),f2(i,j),f2(i,j+1),.5*(h(i,j-1)+h(i,j)))
     *    -vdiv2(f2(i,j),f1(i+1,j),f1(i+1,j-1),f1(i,j-1),f1(i,j),
     *                 .5*(h(i,j-1)+h(i,j)))
      endif
      if(isor.eq.3) then
      do 61 j=2,n2-2
      do 61 i=3,n1-2
   61 v1(i,j)=v1(i,j)     +vcor31(f1(i,j),
     1        x(i-2,j),x(i-1,j),x(i,j),x(i+1,j),.5*(h(i-1,j)+h(i,j)))
      do j=2,n2-2
      v1(2,j)=v1(2,j)     +vcor31(f1(2,j),
     1        x(n1-2,j),x(1,j),x(2,j),x(3,j),.5*(h(1,j)+h(2,j)))
      v1(n1-1,j)=v1(n1-1,j) +vcor31(f1(n1-1,j),x(n1-3,j),x(n1-2,j),
     1                  x(n1-1,j),x(2,j),.5*(h(n1-2,j)+h(n1-1,j)))
      enddo
      do 62 j=2,n2-2
      do 62 i=2,n1-1
   62 v1(i,j)=v1(i,j)
     1 +vcor32(f1(i,j),f2(i-1,j)+f2(i-1,j+1)+f2(i,j+1)+f2(i,j),
     *   abs(x(i,j+1))-abs(x(i,j-1))-abs(x(i-1,j+1))+abs(x(i-1,j-1)),
     *   abs(x(i,j+1))+abs(x(i,j-1))+abs(x(i-1,j+1))+abs(x(i-1,j-1))+ep,
     *                   .5*(h(i-1,j)+h(i,j)))
      do 63 j=3,n2-2
      do 63 i=1,n1-1
   63 v2(i,j)=v2(i,j)     +vcor31(f2(i,j),
     1        x(i,j-2),x(i,j-1),x(i,j),x(i,j+1),.5*(h(i,j-1)+h(i,j)))
      do 64 j=3,n2-2
      do 64 i=2,n1-2
   64 v2(i,j)=v2(i,j)
     1 +vcor32(f2(i,j),f1(i,j-1)+f1(i+1,j-1)+f1(i+1,j)+f1(i,j),
     *   abs(x(i+1,j))-abs(x(i-1,j))-abs(x(i+1,j-1))+abs(x(i-1,j-1)),
     *   abs(x(i+1,j))+abs(x(i-1,j))+abs(x(i+1,j-1))+abs(x(i-1,j-1))+ep,
     *                   .5*(h(i,j-1)+h(i,j)))
      do 641 j=3,n2-2
      v2(1,j)=v2(1,j)
     1 +vcor32(f2(1,j),f1(1,j-1)+f1(2,j-1)+f1(2,j)+f1(1,j),
     *   abs(x(2,j))-abs(x(n1-2,j))-abs(x(2,j-1))+abs(x(n1-2,j-1)),
     *   abs(x(2,j))+abs(x(n1-2,j))+abs(x(2,j-1))+abs(x(n1-2,j-1))+ep,
     *                   .5*(h(1,j-1)+h(1,j)))
  641 v2(n1-1,j)=v2(1,j)
      endif
 
        do j=1,n2m
          v1( 1,j)=v1(n1m,j)
          v1(n1,j)=v1(  2,j)
        end do

      if (iflg.ne.6) then
        do i=1,n1m
          v2(i, 1)=-v2(i,  2)
          v2(i,n2)=-v2(i,n2m)
        end do
      end if

                  if(nonos.eq.1) then
c                 non-osscilatory option
      do 401 j=1,n2m
      jm=max0(j-1,1  )
      jp=min0(j+1,n2m)
      do 401 i=1,n1m
      im=(i-1+(n1-i)/n1m*(n1-2))
      ip=(i+1    -i /n1m*(n1-2))
      mx(i,j)=amax1(x(im,j),x(i,j),x(ip,j),x(i,jm),x(i,jp),mx(i,j))
  401 mn(i,j)=amin1(x(im,j),x(i,j),x(ip,j),x(i,jm),x(i,jp),mn(i,j))

      do 402 j=1,n2m 
      do 4021 i=2,n1-1
 4021 f1(i,j)=donor(x(i-1,j),x(i,j),v1(i,j))
      f1(1 ,j)=f1(n1m,j)
      f1(n1,j)=f1(2  ,j)
  402 continue
     
      do 403 i=1,n1m
      do 4031 j=2,n2m
 4031 f2(i,j)=donor(x(i,j-1),x(i,j),v2(i,j))
      if(iflg.ne.6) then
      f2(i, 1)=-f2(i,  2)
      f2(i,n2)=-f2(i,n2m)
      else
      f2(i, 1)=0.
      f2(i,n2)=0.
      endif
  403 continue

      do 404 j=1,n2m   
      do 404 i=1,n1m
      cp(i,j)=(mx(i,j)-x(i,j))*h(i,j)/
     1(pn(f1(i+1,j))+pp(f1(i,j))+pn(f2(i,j+1))+pp(f2(i,j))+ep)
      cn(i,j)=(x(i,j)-mn(i,j))*h(i,j)/
     1(pp(f1(i+1,j))+pn(f1(i,j))+pp(f2(i,j+1))+pn(f2(i,j))+ep)
  404 continue
      do 405 j=1,n2m 
      do 4051 i=2,n1m 
 4051 v1(i,j)=pp(v1(i,j))*
     1  ( amin1(1.,cp(i,j),cn(i-1,j))*pp(sign(1., x(i-1,j)))
     1   +amin1(1.,cp(i-1,j),cn(i,j))*pp(sign(1.,-x(i-1,j))) )
     2       -pn(v1(i,j))*
     2  ( amin1(1.,cp(i-1,j),cn(i,j))*pp(sign(1., x(i ,j )))
     2   +amin1(1.,cp(i,j),cn(i-1,j))*pp(sign(1.,-x(i ,j ))) )
      v1( 1,j)=v1(n1m,j)
      v1(n1,j)=v1( 2 ,j)
  405 continue

      do 406 i=1,n1m 
      do 406 j=2,n2m 
  406 v2(i,j)=pp(v2(i,j))*
     1  ( amin1(1.,cp(i,j),cn(i,j-1))*pp(sign(1., x(i,j-1)))
     1   +amin1(1.,cp(i,j-1),cn(i,j))*pp(sign(1.,-x(i,j-1))) )
     1       -pn(v2(i,j))*
     2  ( amin1(1.,cp(i,j-1),cn(i,j))*pp(sign(1., x(i ,j )))
     2   +amin1(1.,cp(i,j),cn(i,j-1))*pp(sign(1.,-x(i ,j ))) )
                  endif
    3                      continue
    6 continue
      return

      contains

      real function donor(y1,y2,a)
      real, intent(in) :: y1,y2,a
      if (a .ge. 0.0) then
        donor = y1*a
      else
        donor = y2*a
      endif
      end function

      real function vdyf(x1,x2,a,r)
      real, intent(in) :: x1,x2,a,r
      real, parameter :: ep = 1.e-12
      vdyf = (abs(a)-a*a/r)*(abs(x2)-abs(x1)) / (abs(x2)+abs(x1)+ep)
      end function

      real function vcorr(a,b,y1,y2,r)
      real, intent(in) :: a,b,y1,y2,r
      vcorr = -0.125*a*b*y1/(y2*r)
      end function

      real function vcor31(a,x0,x1,x2,x3,r)
      real a,x0,x1,x2,x3,r
      real ep
      parameter (ep=1.e-12)

      vcor31 = -(a -3.*abs(a)*a/r+2.*a**3/r**2)/3.
     1 *(abs(x0)+abs(x3)-abs(x1)-abs(x2))
     2 /(abs(x0)+abs(x3)+abs(x1)+abs(x2)+ep)

      return
      end function

      real function vcor32(a,b,y1,y2,r)
      real a,b,y1,y2,r

      vcor32 = 0.25*b/r*(abs(a)-2.*a**2/r)*y1/y2

      return
      end function

      real function vdiv1(a1,a2,a3,r)
      real a1,a2,a3,r

      vdiv1 = 0.25*a2*(a3-a1)/r

      return
      end function

      real function vdiv2(a,b1,b2,b3,b4,r)
      real a,b1,b2,b3,b4,r

      vdiv2 = 0.25*a*(b1+b2-b3-b4)/r

      return
      end function

      real function pp(y)
      real y

      pp = amax1(0.0,y)

      return
      end function

      real function pn(y)
      real y

      pn = -amin1(0.0,y)

      return
      end function
      end