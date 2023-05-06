if(!settings.multipleView) settings.batchView=false;
settings.tex="pdflatex";
defaultfilename="main-1";
if(settings.render < 0) settings.render=4;
settings.outformat="";
settings.inlineimage=true;
settings.embed=true;
settings.toolbar=false;
viewportmargin=(2,2);


if(!settings.multipleView) settings.batchView=false;
import three;
settings.tex="pdflatex";
defaultfilename="input-1";
if(settings.render < 0) settings.render=40;
settings.outformat="";
settings.inlineimage=true;
settings.embed=true;
settings.toolbar=false;
defaultpen(fontsize(8pt));



size(9.5cm);
settings.outformat="pdf";
settings.prc=false;
import graph3;
import grid3;

currentprojection=orthographic(
camera=(2.9637389483254,1.48144594554777,1.03154384251566),
up=(-0.00245695858860577,0.00028972293310393,0.00664306468788549),
target=(0,0,0),
zoom=0.971150237157127);


//Draw Axes
pen thickblack = black+0.75;
real axislength = 1.0;
draw(L=Label("$x$", position=Relative(1.1), align=SW),
O--axislength*X,thickblack, Arrow3);
draw(L=Label("$y$", position=Relative(1.1), align=E),
O--axislength*Y,thickblack, Arrow3);
draw(L=Label("$z$", position=Relative(1.1), align=N),
O--axislength*Z,thickblack, Arrow3);

//Set parameters of start corner of polar volume element
real r = 0.8;
real q=0.25pi; //theta
real f=0.3pi; //phi

real dq=0.25; //dtheta
real df=0.25; //dphi
real dr=0.15;

triple A = r*expi(q,f);
triple Ar = (r+dr)*expi(q,f);
triple Aq = r*expi(q+dq,f);
triple Arq = (r+dr)*expi(q+dq,f);
triple Af = r*expi(q,f+df);
triple Arf = (r+dr)*expi(q,f+df);
triple Aqf = r*expi(q+dq,f+df);
triple Arqf = (r+dr)*expi(q+dq,f+df);

//label("$A$",A);
//label("$Ar$",Ar);
//label("$Aq$",Aq);
//label("$Arq$",Arq);
//label("$Af$",Af);
//label("$Aqf$",Aqf);
//label("$Arf$",Arf);
//label("$\; \; \quad Arqf$",Arqf);


pen thingray = gray+0.33;

draw(A--Ar);
draw(Aq--Arq);
draw(Af--Arf);
draw(Aqf--Arqf);
draw(O--Af, dashed);
draw(O--Aqf);
draw( arc(O,A,Aq) ,thickblack );
draw( arc(O,Af,Aqf),thickblack );
draw( arc(O,Ar,Arq) );
draw( arc(O,Arf,Arqf) );
draw( arc(O,Ar,Arq) );
draw( arc(O,A,Af),thickblack );
draw( arc(O,Aq,Aqf),thickblack );
draw( arc(O,Ar,Arf) );
draw( arc(O,Arq,Arqf));

pen thinblack = black+0.25;

//phi arcs
draw(O--expi(pi/2,f),thinblack);
draw("$\phi$", arc(O,0.5*X,0.5*expi(pi/2,f)),thinblack,Arrow3);
draw(O--expi(pi/2,f+df),thinblack);
draw( "$d\phi$", arc(O,expi(pi/2,f),expi(pi/2,f+df) ),thinblack , Arrow3);
draw( A.z*Z -- A,thinblack);
draw(L=Label(rotate(-5)*"$r\sin{\theta}$",position=Relative(0.5),align=N), A.z*Z -- Af,thinblack);

//cotheta arcs
draw( arc(O,Aq,expi(pi/2,f)),thinblack );
draw( arc(O,Aqf,expi(pi/2,f+df) ),thinblack);

//theta arcs
draw(O--A,thinblack);
draw(O--Aq,thinblack);
draw(L=Label("$\theta$",position=Relative(0.5), align=NE),
arc(O,0.25*length(A)*Z,0.25*A),thinblack,Arrow3);
draw(L=Label("$d\theta$",position=Relative(0.5),align=NE) ,arc(O,0.66*A,0.66*Aq),
thinblack, Arrow3 );


// inner surface
triple rin(pair t) { return r*expi(t.x,t.y);}
triple rout(pair t) { return 1.24*r*expi(t.x,t.y);}
surface inner=surface(rin,(q,f),(q+dq,f+df),16,16);
draw(inner,emissive(gray+opacity(0.33)));

// surface sider=surface(rin,(f,r),(f+df,r+dr),16,16);
// draw(sider,emissive(gray+opacity(0.33)));


//part of a nearly transparent sphere to help see perspective
surface sphere=surface(rout,(0,0),(pi/2,pi/2),26,26);
draw(sphere,emissive(gray+opacity(0.125)));


// dr and rdtheta labels
triple V= Af + 0.5*(Arf-Af);
draw(L=Label("$dr$",position=Relative(1.1)), V--(1.5*V.x,1.5*V.y,V.z),dotted);
triple U=expi(q+0.5*dq,f);
draw(L=Label("$rd\theta$",position=Relative(1.1)),
r*U ---r*(1.66*U.x,1.66*U.y,U.z),dotted );
