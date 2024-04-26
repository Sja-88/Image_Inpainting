/**
 * A best exemplar finder.  Scans over the entire image (using a
 * sliding window) and finds the exemplar which minimizes the sum
 * squared error (SSE) over the to-be-filled pixels in the target
 * patch. 
 *
 * @author Sooraj Bhat
 */
#include "mex.h"
#include <limits.h>

void searchexemplarhelper(const int mm, const int nn, const int m, const int n, 
			const double *img, const double *Ip, 
			const mxLogical *toFill, const mxLogical *sourceRegion,
			double *dSSD, double *indexx, double *indexy, double *dd) 
{
  register int s,i,j,ii,jj,ii2,jj2,M,N,I,J,ndx,ndx2,mn=m*n,mmnn=mm*nn;
  double patchErr=0.0,err=0.0;
  s=0;

  /* foreach patch */
  N=nn-n+1;  M=mm-m+1;
  for (j=1; j<=N; ++j) {
    J=j+n-1;
    for (i=1; i<=M; ++i) {
      I=i+m-1;
      /*** Calculate patch error ***/
      /* foreach pixel in the current patch */
      for (jj=j,jj2=1; jj<=J; ++jj,++jj2) {
	    for (ii=i,ii2=1; ii<=I; ++ii,++ii2) {
	      ndx=ii-1+mm*(jj-1);
	      if (!sourceRegion[ndx]) goto skipPatch;
	      ndx2=ii2-1+m*(jj2-1);
	      if (!toFill[ndx2]) {
	        err=img[ndx      ] - Ip[ndx2    ]; patchErr += err*err;
	        err=img[ndx+=mmnn] - Ip[ndx2+=mn]; patchErr += err*err;
	        err=img[ndx+=mmnn] - Ip[ndx2+=mn]; patchErr += err*err;
	      }
	    }
      }
      /*** Update ***/
      dSSD[s]=patchErr;
      indexx[s]=i;
      indexy[s]=j;
      s=s+1;
      /*** Reset ***/
      skipPatch:
          patchErr = 0.0; 
    }
  }
  *dd=s;
}

/* [dSSD,indexx,indexy,dd] = searchexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion); */
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]) 
{
  int mm,nn,m,n;
  double *img,*Ip,*dSSD,*indexx,*indexy,*dd;
  mxLogical *toFill,*sourceRegion;

  /* Extract the inputs */
  mm = (int)mxGetScalar(prhs[0]);
  nn = (int)mxGetScalar(prhs[1]);
  m  = (int)mxGetScalar(prhs[2]);
  n  = (int)mxGetScalar(prhs[3]);
  img = mxGetPr(prhs[4]);
  Ip  = mxGetPr(prhs[5]);
  toFill = mxGetLogicals(prhs[6]);
  sourceRegion = mxGetLogicals(prhs[7]);
  
  /* Setup the output */
  plhs[0] = mxCreateDoubleMatrix(500000,1,mxREAL);
  dSSD = mxGetPr(plhs[0]);
  
  plhs[1] = mxCreateDoubleMatrix(500000,1,mxREAL);
  indexx = mxGetPr(plhs[1]);
  
  plhs[2] = mxCreateDoubleMatrix(500000,1,mxREAL);
  indexy = mxGetPr(plhs[2]);
  
  plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
  dd = mxGetPr(plhs[3]);

  /* Do the actual work */
  searchexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion,dSSD,indexx,indexy,dd);
}
