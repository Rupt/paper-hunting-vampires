C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,1)
C     
      SUBROUTINE FFV1smeVEUP_0(F1, F2, V3, COUP,VERTEX)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(*)
      COMPLEX*16 F2(*)
      COMPLEX*16 TMPV
      COMPLEX*16 V3(*)
      COMPLEX*16 VERTEX

      COMPLEX*16 tmp0
      COMPLEX*16 tmp1
      COMPLEX*16 tmp2
      COMPLEX*16 tmp3
      COMPLEX*16 tmp8
      COMPLEX*16 tmp9
      COMPLEX*16 tmp10
      COMPLEX*16 tmp11

      COMPLEX*16 cv00
      COMPLEX*16 cv01
      COMPLEX*16 cv02
      COMPLEX*16 cv03
      COMPLEX*16 cv10
      COMPLEX*16 cv11
      COMPLEX*16 cv12
      COMPLEX*16 cv13
      COMPLEX*16 cv20
      COMPLEX*16 cv21
      COMPLEX*16 cv22
      COMPLEX*16 cv23
      COMPLEX*16 cv30
      COMPLEX*16 cv31
      COMPLEX*16 cv32
      COMPLEX*16 cv33

C   Define the sme coefficients

      cv00 = (0D0,0D0)
      cv01 = (-0.79D0,-0.43D0)
      cv02 = (0D0,0D0)
      cv03 = (0D0,0D0)
      cv10 = (-0.79D0,0.43D0)
      cv11 = (0D0,0D0)
      cv12 = (0.06D0,0.18D0)
      cv13 = (0D0,0D0)
      cv20 = (0D0,-0.0D0)
      cv21 = (0.06D0,-0.18D0)
      cv22 = (0D0,0D0)
      cv23 = (0D0,0D0)
      cv30 = (0D0,-0.0D0)
      cv31 = (0D0,-0.0D0)
      cv32 = (0D0,-0.0D0)
      cv33 = (0D0,0D0)

C   Define some tmp variables

      tmp0 = F1(6)*F2(3) + F1(5)*F2(4) - F1(4)*F2(5) - F1(3)*F2(6)
      tmp1 = F1(6)*F2(3) - F1(5)*F2(4) - F1(4)*F2(5) + F1(3)*F2(6)
      tmp2 = F1(5)*F2(3) - F1(6)*F2(4) - F1(3)*F2(5) + F1(4)*F2(6)
      tmp3 = F1(5)*F2(3) + F1(6)*F2(4) + F1(3)*F2(5) + F1(4)*F2(6)

      tmp8=cv00*V3(3)+cv01*V3(4)+cv02*V3(5)+cv03*V3(6)
      tmp9=cv10*V3(3)+cv11*V3(4)+cv12*V3(5)+cv13*V3(6)
      tmp10=cv20*V3(3)+cv21*V3(4)+cv22*V3(5)+cv23*V3(6)
      tmp11=cv30*V3(3)+cv31*V3(4)+cv32*V3(5)+cv33*V3(6)

      TMPV = (0.5D0)*(tmp0*tmp9 + tmp1*(-CI)*tmp10 + tmp2*tmp11 + tmp3*tmp8)

      VERTEX = COUP*(-CI * TMPV)
      END