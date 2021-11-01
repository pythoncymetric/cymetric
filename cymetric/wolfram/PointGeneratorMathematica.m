(* ::Package:: *)

SamplePointsOnSphere[dimP_, numPts_] := Module[{randomPoints}, (
randomPoints=RandomVariate[NormalDistribution[], {numPts, dimP, 2}];
    randomPoints=randomPoints[[;;,;;,1]] + I randomPoints[[;;,;;,2]];
    randomPoints = Normalize /@ randomPoints;
    Return[randomPoints];
)];

getPointsOnCY[varsUnflat_,numParamsInPn_,dimPs_,params_,pointsOnSphere_,eqns_,precision_:20]:= Module[{subst, pts, i, j, a, b, res, maxPoss, absPts}, ( 
    subst={};
    pts={};
    For[j=1,j<=Length[dimPs],j++,
     AppendTo[subst,Table[varsUnflat[[j, a]]->Sum[params[[j,b]] pointsOnSphere[[j,b,a]],{b,Length[params[[j]]]}],{a,Length[varsUnflat[[j]]]}]];
];
    subst=Flatten[subst];
    (*Print[pointsOnSphere];Print[subst];*)
    (*res=Quiet[Solve[Table[eqns[[i]]==0,{i,Length[eqns]}]/.subst]];*)
    
    res=FindInstance[Table[eqns[[i]]==0, {i,Length[eqns]}]/.subst,Variables[Flatten[params]],Complexes,1000,WorkingPrecision->precision];
    pts=Chop[(varsUnflat/.subst)/.res];
    (*go to patch where largest coordinate is 1*)
absPts=Abs[pts];
    For[i=1,i<=Length[pts],i++,
     pts[[i]]=Chop[Flatten[Table[pts[[i,j]]/pts[[i,j,Ordering[absPts[[i,j]],-1][[1]]]],{j,Length[dimPs]}]]];
];
    Return[pts];
    )];
    
PrintMsg[msg_,frontEnd_,verbose_]:=Module[{},(
If[verbose>0,
If[frontEnd,
Print[msg];
,
ClientLibrary`SetInfoLogLevel[];
ClientLibrary`info[msg];
ClientLibrary`SetErrorLogLevel[];
];
];
)];

GeneratePointsM[numPts_, dimPs_, coefficients_, exponents_, precision_:20,verbose_:0,frontEnd_:False]:=Module[{varsUnflat,vars,eqns,i,j,conf,start,col,totalDeg,numParamsInPn,numPoints,ptsPartition,params,pointsOnSphere,pointsOnCY,numPtsPerSample},( 
    varsUnflat=Table[Subscript[x, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
    vars=Flatten[varsUnflat];
    (*Reconstruct equations*)
    eqns=Table[Sum[coefficients[[i,j]] Times@@(Power[vars,exponents[[i,j]]]),{j,Length[coefficients[[i]]]}],{i,Length[coefficients]}];
(*Reconstruct the transpose configuration matrix / multi-degrees of each equation*)
    conf= {};
    For[i=1,i<=Length[coefficients],i++,
start=1;
col={};
For[j=1,j<=Length[dimPs],j++,
totalDeg=Plus@@exponents[[i,1,start;;start+dimPs[[j]]]];
AppendTo[col,totalDeg];
      start+=dimPs[[j]]+1;
      ];
AppendTo[conf, col];
];
PrintMsg["Configuration matrix: "<>ToString[Transpose[conf]],frontEnd,verbose];

    (*Find lowest degree in each equation while ensuring that each equation gets at least one parameter*)
    (*Need to get points upon intersection with equations, i.e. ] we need as many parameters as equations*)
    (*We want the degree in the parameters to be as small as possible, while at the same time ensuring that each equation has at least one parameter such that it can be solved. Instead of finding the optimal configuration for this, we content ourselfs with finding a good one (which can be found much faster)*)
    (*In a first pass, make sure that each equation gets a parameter*)
    numParamsInPn=Table[0,{i,Length[dimPs]}];
    For[i=1,i<=Length[eqns],i++,
If[Union[numParamsInPn*conf[[i]]]=={0},
numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
];
];

(*Now make sure that we have as many parameters as equations (we have at most as many atm.)*)
    While[Length[eqns]!=Plus@@numParamsInPn,
     For[i=1,i<=Length[eqns],i++,
If[Length[eqns]==Plus @@ numParamsInPn, Break[];];
numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
];
     ];

(*Finally, we make sure that there are at most n parameters in each P^n*)
i=1;
While[i<=Length[numParamsInPn],
If[numParamsInPn[[i]]>dimPs[[i]],
For[j=1,j<=Length[numParamsInPn],j++,
If[numParamsInPn[[j]]>=dimPs[[j]],
Continue[];,
numParamsInPn[[j]]++; numParamsInPn[[i]]--; Break[];];
];
Continue[];
];
i++;
];
PrintMsg["Number of Parameters per P^n: "<>ToString[numParamsInPn],frontEnd,verbose];
    
(*Generate points on CY. Do one trial run to find how many points you get from one intersection *)
numPoints=1;
Clear[t];
params=Table[Join[{1},Table[Subscript[t,j,k],{k,numParamsInPn[[j]]}]],{j,Length[numParamsInPn]}];
pointsOnSphere=ParallelTable[SamplePointsOnSphere[dimPs[[i]]+1,numPoints (numParamsInPn[[i]]+1)],{i,Length[dimPs]}];

(*Create system of equations and solve it to find points on CY*)
pointsOnCY=ParallelTable[getPointsOnCY[varsUnflat, numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns,precision],{p,numPoints}];
pointsOnCY=Flatten[pointsOnCY,1];
numPtsPerSample=Length[pointsOnCY];
PrintMsg["Number of points on CY from one ambient space intersection: "<>ToString[numPtsPerSample],frontEnd,verbose];
    
(*Now generate as many points as needed*)
numPoints=Ceiling[numPts/numPtsPerSample];
PrintMsg["Now generating "<>ToString[numPts]<>" points...",frontEnd,verbose];

Clear[x,t];
varsUnflat=Table[Subscript[x,i,a],{i,Length[dimPs]},{a,0,dimPs[[i]]}];
params=Table[Join[{1},Table[Subscript[t,j,k],{k,numParamsInPn[[j]]}]],{j,Length[numParamsInPn]}];
pointsOnSphere=ParallelTable[SamplePointsOnSphere[dimPs[[i]]+1,numPoints (numParamsInPn[[i]]+1)],{i,Length[dimPs]}];
    
(*Create system of equations and solve it to find points on CY*)
If[frontEnd,
    pointsOnCY=ResourceFunction["MonitorProgress"][ParallelTable[getPointsOnCY[varsUnflat,numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns],{p,numPoints}]];
    ,
    If[verbose==0,
    pointsOnCY=ParallelTable[getPointsOnCY[varsUnflat,numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns],{p,numPoints}];
    ,
    (*Partition in order to provide progress feedback (WolframClient Library ignores messages from subkernels spawned from the kernel used in wl.evaluate(). This negatively impacts performance)*)
    pointsOnCY={};
    For[i=1,i<=20,i++,
    PrintMsg["Generated "<>ToString[5 (i-1)]<>"% of points",frontEnd,verbose];
    pointsOnCY=Join[pointsOnCY,ParallelTable[getPointsOnCY[varsUnflat,numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns],{p,Ceiling[numPoints/20]}]];
    ];
    ];
];
PrintMsg["done.",frontEnd,verbose];
pointsOnCY=Flatten[pointsOnCY,1];
Return[pointsOnCY];
)];

GetPointsOnCYToric[CYEqn_,vars_,sections_,patchMasks_,sectionMonoms_,GLSMcharges_,precision_]:=Module[{i,j,k,l,eq,dimPs,patchMask,numEqnsInPn,tmpMask,coeffs,newEqn,pts,patchCoords,scalings,scaleSol,tmpPts,lambdas},( 
patchMask=RandomChoice[patchMasks];
eq={CYEqn};
dimPs=Table[Length[sections[[i]]],{i,Length[sections]}];
numEqnsInPn=Table[0,{i,Length[dimPs]}];
(*Need to get points upon intersection with equations i.e. we need to find the right number of sections. We have len(dimPs) many scalings (assume favorable, simplicial KC case), and one hypersurface equation, so we need dimCY many conditions *)
For[i=1,i<=Length[patchMask],i++,
(* patch coordinate \[Rule] set coordinate to one *)
If[patchMask[[i]]==1,
AppendTo[eq,vars[[i]]==1.];
Continue[];
];
If[Plus@@numEqnsInPn==Length[vars]-Length[sections]-1,Continue[];];
(*else search through sections and take the first one where the coordinate appears subject to the constraint that we do not overshoot the dimensionality*)
For[j=1,j<=Length[sections],j++,
(*already enough equations in this P^n*)
If[numEqnsInPn[[j]]>= dimPs[[j]],Continue[];];
tmpMask=Plus@@sections[[j]];
If[tmpMask[[i]]!=0,(*coordinate appears in this Pn*)
coeffs=RandomVariate[NormalDistribution[],{Length[sectionMonoms[[j]]],2}];
newEqn=Sum[(coeffs[[k,1]]+I coeffs[[k,2]])sectionMonoms[[j,k]] ,{k,Length[coeffs]}];
AppendTo[eq,newEqn==0];
numEqnsInPn[[j]]+=1;
Break[];
];
];
];
pts=Quiet[vars/.NSolve[eq,vars,WorkingPrecision->precision]];
Clear[\[Lambda]];
lambdas=Table[\[Lambda][k],{k,Length[GLSMcharges]}];
(*go to patch where largest coordinate is 1*)
For[l=1,l<=Length[pts],l++,
For[i=1,i<=Length[patchMasks],i++,(*try all patches*)
patchCoords=Flatten[Position[patchMasks[[i]],1]];(*find coordinates that are non-zero => can be scaled to 1*)
eq={};
scalings=Table[1,{k,Length[pts[[1]]]}];
For[k=1,k<=Length[GLSMcharges],k++,(*perform scaling according to GLSM charges*)
scalings*=lambdas[[k]]^GLSMcharges[[k]];
];
eq=Table[1==pts[[l,patchCoords[[j]]]]*scalings[[patchCoords[[j]]]],{j,Length[patchCoords]}];
scaleSol=Quiet[Solve[eq]];
If[Length[scaleSol]>0,
tmpPts=(scalings*pts[[l]])/.scaleSol[[1]];
];
If[Max[Abs[tmpPts]]==1,
pts[[l]]=Chop[tmpPts];
Break[];
];
];
];
Return[pts];
)];

GenerateToricPointsM[numPts_,dimCY_,coefficients_,exponents_,sections_,patchMasks_,GLSMcharges_,precision_:20,verbose_:0,frontEnd_:False]:= Module[{vars,CYeqn,i,j,k,sectionMonoms,numPoints,params,allEqns,pointsOnCY,newPoints,numPtsPerSample},( 
vars=Table[Subscript[x,i],{i,Length[sections]+dimCY+1}];
(*Reconstruct equations*)
CYeqn=(0.==Sum[coefficients[[i]] Times@@(vars^exponents[[i]]),{i,Length[coefficients]}]);
(*Reconstruct sections*)
sectionMonoms=Table[Table[Times@@(vars^sections[[i,j]]),{j,Length[sections[[i]]]}],{i,Length[sections]}];
(*Generate points on CY.Do one trial run to find how many points you get from one intersection*)
pointsOnCY=ParallelTable[GetPointsOnCYToric[CYeqn,vars,sections,patchMasks,sectionMonoms,GLSMcharges,precision],{p,10}];
numPtsPerSample=Min[Table[Length[pointsOnCY[[i]]],{i,Length[pointsOnCY]}]];
pointsOnCY=Flatten[pointsOnCY,1];
PrintMsg["Number of points on CY from one ambient space intersection: "<>ToString[numPtsPerSample],frontEnd,verbose];

(*Now generate as many points as needed*)
numPoints=Ceiling[(numPts-Length[pointsOnCY])/numPtsPerSample];
PrintMsg["Now generating "<>ToString[numPts]<>" points...",frontEnd,verbose];

(*Create system of equations and solve it to find points on CY*)
If[frontEnd,
    newPoints=ResourceFunction["MonitorProgress"][ParallelTable[GetPointsOnCYToric[CYeqn,vars,sections,patchMasks,sectionMonoms,GLSMcharges,precision],{p,numPoints}]];
    ,
    If[verbose==0,
    newPoints=ResourceFunction["MonitorProgress"][ParallelTable[GetPointsOnCYToric[CYeqn,vars,sections,patchMasks,sectionMonoms,GLSMcharges,precision],{p,numPoints}]];
    ,
    (*Partition in order to provide progress feedback (WolframClient Library ignores messages from subkernels spawned from the kernel used in wl.evaluate(). This negatively impacts performance)*)
    newPoints={};
    For[i=1,i<=20,i++,
    PrintMsg["Generated "<>ToString[5 (i-1)]<>"% of points",frontEnd,verbose];
    newPoints=Join[newPoints,ParallelTable[GetPointsOnCYToric[CYeqn,vars,sections,patchMasks,sectionMonoms,GLSMcharges,precision],{p,Ceiling[numPoints/20]}]];
    ];
    ];
];
pointsOnCY=Join[pointsOnCY,Flatten[newPoints,1]];
If[Length[pointsOnCY]>numPts,pointsOnCY=pointsOnCY[[1;;numPts]]];
PrintMsg["done.",frontEnd,verbose];
Return[pointsOnCY];
)];
