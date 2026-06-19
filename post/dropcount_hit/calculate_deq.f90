subroutine calculate_deq(vol_cells, nstep, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
  use param
  implicit none
  integer, intent(in)          :: vol_cells, nstep
  double precision, intent(in) :: Ixx, Iyy, Izz, Ixy, Ixz, Iyz
  character(len=clen)          :: time, namefile
  double precision             :: vol, deq, eig(3)

  vol = dble(vol_cells)

  ! Equivalent diameter: twice the radius of the equivalent sphere (V = (4/3)*pi*R^3)
  deq = dx * (n6opi * vol)**(1.0d0/3.0d0)

  call eig3x3_sym(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, eig)

  write(*,'(1x,a,E16.6)')      'Diameter   ', deq
  write(*,'(1x,a,3E16.6)')     'Eigenvalues', eig(1), eig(2), eig(3)
  write(*,*)

  write(time,'(i8.8)') nstep
  namefile = 'deq_'//trim(time)//'.dat'
  open(3, file=trim(namefile), access='append', form='formatted', status='unknown')
    write(3,'(E12.5)') deq
  close(3, status='keep')

  ! One row per droplet: Ixx Iyy Izz Ixy Ixz Iyz  lambda1 lambda2 lambda3
  namefile = 'inertia_'//trim(time)//'.dat'
  open(4, file=trim(namefile), access='append', form='formatted', status='unknown')
    write(4,'(9E16.6)') Ixx, Iyy, Izz, Ixy, Ixz, Iyz, eig(1), eig(2), eig(3)
  close(4, status='keep')

end subroutine calculate_deq


! Eigenvalues of the symmetric 3x3 matrix
!   [ a  b  c ]
!   [ b  d  e ]
!   [ c  e  f ]
! returned in ascending order in eig(1:3).
! Uses the trigonometric (Cardano) method from Smith (1961) / Wikipedia.
subroutine eig3x3_sym(a, b, c, d, e, f, eig)
  use param, only: pi
  implicit none
  double precision, intent(in)  :: a, b, c, d, e, f
  double precision, intent(out) :: eig(3)
  double precision :: q, p1, p2, p, r, phi
  double precision :: B11, B12, B13, B22, B23, B33, tmp

  p1 = b*b + c*c + e*e

  if (p1 == 0.0d0) then
    eig(1) = a;  eig(2) = d;  eig(3) = f
  else
    q  = (a + d + f) / 3.0d0
    p2 = (a-q)**2 + (d-q)**2 + (f-q)**2 + 2.0d0*p1
    p  = sqrt(p2 / 6.0d0)

    B11 = (a - q) / p;  B12 = b / p;  B13 = c / p
    B22 = (d - q) / p;  B23 = e / p
    B33 = (f - q) / p

    r = 0.5d0 * (B11*(B22*B33 - B23*B23) &
               - B12*(B12*B33 - B23*B13) &
               + B13*(B12*B23 - B22*B13))
    r = max(-1.0d0, min(1.0d0, r))   ! guard against round-off outside [-1,1]

    phi = acos(r) / 3.0d0

    eig(3) = q + 2.0d0*p*cos(phi)
    eig(1) = q + 2.0d0*p*cos(phi + 2.0d0*pi/3.0d0)
    eig(2) = 3.0d0*q - eig(3) - eig(1)
  end if

  ! Sort ascending (3-element bubble sort)
  if (eig(1) > eig(2)) then;  tmp = eig(1);  eig(1) = eig(2);  eig(2) = tmp;  end if
  if (eig(2) > eig(3)) then;  tmp = eig(2);  eig(2) = eig(3);  eig(3) = tmp;  end if
  if (eig(1) > eig(2)) then;  tmp = eig(1);  eig(1) = eig(2);  eig(2) = tmp;  end if

end subroutine eig3x3_sym
