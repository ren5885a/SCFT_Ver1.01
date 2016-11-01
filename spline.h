/*----------------------------------------------------------------------------------------------*/
/* spline.c
   Cubic interpolating spline. 
   from http://www.mech.uq.edu.au/staff/jacobs/nm_lib/cmathsrc
*/

/************************************************/
/*                                              */
/*  CMATH.  Copyright (c) 1989 Design Software  */
/*                                              */
/************************************************/

/*----------------------------------------------------------------------------------------------*/
/* spline.c
   Cubic interpolating spline.  from http://www.mech.uq.edu.au/staff/jacobs/nm_lib/cmathsrc
*/   
/* Cubic spline coefficients */
int 	spline (int n, int e1, int e2, double s1, double s2, double x[], double y[], double b[], double c[], double d[], int *flag);
/* spline evaluation */
double 	seval (int n, double xx, double x[], double y[], double b[], double c[], double d[], int *last);
/* derivative evaluation */
double 	deriv (int n, double xx, double x[], double b[], double c[], double d[], int *last);
/* integral of spline */
double 	sinteg (int n, double u, double x[], double y[], double b[], double c[], double d[], int *last);

/*----------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------*/
