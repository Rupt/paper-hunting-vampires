C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,1)
C     
      SUBROUTINE FFV1smeVEDN_1(F2, V3, COUP, M1, W1,F1)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(6)
      COMPLEX*16 F2(*)
      REAL*8 M1
      REAL*8 P1(0:3)
      COMPLEX*16 V3(*)
      REAL*8 W1
      COMPLEX*16 DENOM

      COMPLEX*16 tmp8
      COMPLEX*16 tmp9
      COMPLEX*16 tmp10
      COMPLEX*16 tmp11
      COMPLEX*16 tmp16
      COMPLEX*16 tmp17
      COMPLEX*16 tmp18
      COMPLEX*16 tmp19

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

      F1(1) = +F2(1)+V3(1)
      F1(2) = +F2(2)+V3(2)
      P1(0) = -DBLE(F1(1))
      P1(1) = -DBLE(F1(2))
      P1(2) = -DIMAG(F1(2))
      P1(3) = -DIMAG(F1(1))

C   Define some tmp variables

      tmp8=cv00*V3(3)+cv01*V3(4)+cv02*V3(5)+cv03*V3(6)
      tmp9=cv10*V3(3)+cv11*V3(4)+cv12*V3(5)+cv13*V3(6)
      tmp10=cv20*V3(3)+cv21*V3(4)+cv22*V3(5)+cv23*V3(6)
      tmp11=cv30*V3(3)+cv31*V3(4)+cv32*V3(5)+cv33*V3(6)

      tmp16=P1(0)+P1(3)
      tmp17=P1(0)-P1(3)
      tmp18=P1(1)+CI*P1(2)
      tmp19=P1(1)-CI*P1(2)


      DENOM = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI
     $ * W1))
        
      F1(3) = DENOM*(0.5D0)*( 
     $ ((-CI)*tmp9*tmp18 - tmp10*tmp18 - (CI)*tmp8*tmp16 - (CI)*tmp11*tmp16)*F2(3)
     $ +((-CI)*tmp8*tmp18+ (CI)*tmp11*tmp18 - (CI)*tmp9*tmp16 + tmp10*tmp16)*F2(4)
     $ + M1*(CI)*(tmp8 - tmp11)*F2(5)+ M1*((-CI)*tmp9 + tmp10)*F2(6))

      F1(4) = DENOM*(0.5D0)*(
     $ ((-CI)*tmp8*tmp19 - (CI)*tmp11*tmp19 - (CI)*tmp9*tmp17-tmp10*tmp17)*F2(3)
     $ +(tmp10*tmp19 - (CI)*tmp9*tmp19 - (CI)*tmp8*tmp17+ (CI)*tmp11*tmp17)*F2(4)
     $ + M1*((-CI)*tmp9-tmp10)*F2(5)+  M1*(CI)*(tmp8 + tmp11)*F2(6))

      F1(5) = DENOM*(0.5D0)*(
     $ M1*(CI)*(tmp8 + tmp11)*F2(3) + M1*((CI)*tmp9-tmp10)*F2(4)
     $ + (-tmp10*tmp18 - (CI)*tmp9*tmp18- (CI)*tmp8*tmp17 + (CI)*tmp11*tmp17)*F2(5)
     $ +((CI)*tmp8*tmp18 + (CI)*tmp11*tmp18 + (CI)*tmp9*tmp17 - tmp10*tmp17)*F2(6))

      F1(6) = DENOM*(0.5D0)*(
     $ M1*((CI)*tmp9+tmp10)+F2(3) + M1*(CI)*(tmp8-tmp11)*F2(4)
     $ +((CI)*tmp8*tmp19 - (CI)*tmp11*tmp19 + (CI)*tmp9*tmp16 + tmp10*tmp16)*F2(5)
     $ +((-CI)*tmp9*tmp19 + tmp10*tmp19 - (CI)*tmp8*tmp16 - (CI)*tmp11*tmp16)*F2(6))

      END



      SUBROUTINE FFV1smeVEDNP1N_1(F2, V3, COUP,F1)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(6)
      COMPLEX*16 F2(*)
      COMPLEX*16 V3(*)

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

C   Define some tmp variables

      tmp8=cv00*V3(3)+cv01*V3(4)+cv02*V3(5)+cv03*V3(6)
      tmp9=cv10*V3(3)+cv11*V3(4)+cv12*V3(5)+cv13*V3(6)
      tmp10=cv20*V3(3)+cv21*V3(4)+cv22*V3(5)+cv23*V3(6)
      tmp11=cv30*V3(3)+cv31*V3(4)+cv32*V3(5)+cv33*V3(6)

      F1(3)= COUP*(0.5D0)*CI*((tmp11-tmp8)*F2(5)+(tmp9+CI*tmp10)*F2(6))
      F1(4)= COUP*(0.5D0)*CI*((tmp9-CI*tmp10)*F2(5)-(tmp8+tmp11)*F2(6))
      F1(5)= COUP*(0.5D0)*CI*((-tmp9-CI*tmp10)*F2(4)-(tmp8+tmp11)*F2(3))
      F1(6)= COUP*(0.5D0)*CI*((-tmp9+CI*tmp10)*F2(3)+(tmp11-tmp8)*F2(4))

      END





