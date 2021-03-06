C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,1)
C     
      SUBROUTINE FFV1smeVEDNP0_3(F1, F2, COUP, M3, W3,V3)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(*)
      COMPLEX*16 F2(*)
      REAL*8 M3
      REAL*8 P3(0:3)
      COMPLEX*16 V3(6)
      REAL*8 W3
      COMPLEX*16 DENOM


      COMPLEX*16 tmp24
      COMPLEX*16 tmp25
      COMPLEX*16 tmp26
      COMPLEX*16 tmp27

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
      cv01 = (0.17D0,-0.5D0)
      cv02 = (0D0,0D0)
      cv03 = (0D0,0D0)
      cv10 = (0.17D0,0.5D0)
      cv11 = (0D0,0D0)
      cv12 = (0.9D0,-0.91D0)
      cv13 = (0D0,0D0)
      cv20 = (0D0,-0.0D0)
      cv21 = (0.9D0,0.91D0)
      cv22 = (0D0,0D0)
      cv23 = (0D0,0D0)
      cv30 = (0D0,-0.0D0)
      cv31 = (0D0,-0.0D0)
      cv32 = (0D0,-0.0D0)
      cv33 = (0D0,0D0)


      V3(1) = +F1(1)+F2(1)
      V3(2) = +F1(2)+F2(2)
      P3(0) = -DBLE(V3(1))
      P3(1) = -DBLE(V3(2))
      P3(2) = -DIMAG(V3(2))
      P3(3) = -DIMAG(V3(1))


      tmp24 = cv00*F1(3)-cv01*F1(4)-cv02*F1(5)-cv03*F1(6)
      tmp25= cv10*F1(3)-cv11*F1(4)-cv12*F1(5)-cv13*F1(6)
      tmp26= cv20*F1(3)-cv21*F1(4)-cv22*F1(5)-cv23*F1(6)
      tmp27= cv30*F1(3)-cv31*F1(4)-cv32*F1(5)-cv33*F1(6)

      DENOM = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI
     $ * W3))
   
      V3(3) = DENOM*(0.5D0)*CI*(tmp26*F2(3)+tmp27*F2(4)+tmp24*F2(5)
     $+tmp25*F2(6))

      V3(4) = DENOM*(0.5D0)*CI*(tmp27*F2(3)+tmp26*F2(4)-tmp25*F2(5)
     $-tmp24*F2(6))

      V3(5) = DENOM*(0.5D0)*CI*(-CI*tmp27*F2(3)+CI*tmp26*F2(4)
     $+CI*tmp25*F2(5)-CI*tmp24*F2(6))

      V3(6) = DENOM*(0.5D0)*CI*(tmp26*F2(3)-tmp27*F2(4)-tmp24*F2(5)
     $+tmp25*F2(6))


      END


      SUBROUTINE FFV1smeVEDNP1N_3(F1, F2, COUP,V3)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(*)
      COMPLEX*16 F2(*)
      COMPLEX*16 V3(6)

      COMPLEX*16 tmp0
      COMPLEX*16 tmp1
      COMPLEX*16 tmp2
      COMPLEX*16 tmp3

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
      cv01 = (0.17D0,-0.5D0)
      cv02 = (0D0,0D0)
      cv03 = (0D0,0D0)
      cv10 = (0.17D0,0.5D0)
      cv11 = (0D0,0D0)
      cv12 = (0.9D0,-0.91D0)
      cv13 = (0D0,0D0)
      cv20 = (0D0,-0.0D0)
      cv21 = (0.9D0,0.91D0)
      cv22 = (0D0,0D0)
      cv23 = (0D0,0D0)
      cv30 = (0D0,-0.0D0)
      cv31 = (0D0,-0.0D0)
      cv32 = (0D0,-0.0D0)
      cv33 = (0D0,0D0)

      tmp0 =F1(6)*F2(3) + F1(5)*F2(4) - F1(4)*F2(5) - F1(3)*F2(6)
      tmp1 =F1(6)*F2(3) - F1(5)*F2(4) - F1(4)*F2(5) + F1(3)*F2(6)
      tmp2 = F1(5)*F2(3) - F1(6)*F2(4) - F1(3)*F2(5) + F1(4)*F2(6)
      tmp3 = F1(5)*F2(3) + F1(6)*F2(4) + F1(3)*F2(5) + F1(4)*F2(6)


      V3(3)= COUP*(-0.5D0)*(CI)*(cv10*tmp0-CI*cv20*tmp1+cv30*tmp2+cv00*tmp3)
      V3(4)= COUP*(-0.5D0)*(CI)*(cv11*tmp0-CI*cv21*tmp1+cv31*tmp2+cv01*tmp3)
      V3(5)= COUP*(-0.5D0)*(CI)*(cv12*tmp0-CI*cv22*tmp1+cv32*tmp2+cv02*tmp3)
      V3(6)= COUP*(-0.5D0)*(CI)*(cv13*tmp0-CI*cv23*tmp1+cv33*tmp2+cv03*tmp3)
      END









