(* ::Package:: *)

SamplePointsOnSphere[dimP_, numPts_] := Module[{randomPoints}, (
randomPoints=RandomVariate[NormalDistribution[], {numPts, dimP, 2}];
    randomPoints=randomPoints[[;;,;;,1]] + I randomPoints[[;;,;;,2]];
    randomPoints = Normalize /@ randomPoints;
    Return[randomPoints];
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

GeneratePointsM[numPts_, dimPs_, coefficients_, exponents_, precision_:20,verbose_:0,frontEnd_:False]:=Module[{varsUnflat,vars,eqns,i,j,conf,start,col,totalDeg,numParamsInPn,numPoints,ptsPartition,params,low,pointsOnSphere,pointsOnCY,numPtsPerSample},( 
    varsUnflat=Table[Subscript[x, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
    vars=Flatten[varsUnflat];
    (*Reconstruct equations*)
    eqns=Table[Sum[coefficients[[i,j]] Times@@(Power[vars,exponents[[i,j]]]),{j,Length[coefficients[[i]]]}],{i,Length[coefficients]}];
    (*Print[eqns];*)
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
    (*Need to get points upon intersection with equations, i.e. we need as many parameters as equations*)
    (*We want the degree in the parameters to be as small as possible, while at the same time ensuring that each equation has at least one parameter such that it can be solved. Instead of finding the optimal configuration for this, we content ourselfs with finding a good one (which can be found much faster)*)
    (*In a first pass, make sure that each equation gets a parameter*)
    numParamsInPn=Table[1,{i,Length[dimPs]}];
    For[i=1,i<=Length[eqns]-Length[dimPs],i++,
If[Union[numParamsInPn*conf[[i]]]=={0},
numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
];
];

(*Now make sure that we have as many parameters as equations*)
While[Length[eqns]>Plus@@numParamsInPn,
     For[i=1,i<=Length[eqns],i++,
     If[Length[eqns]==Plus @@ numParamsInPn, Break[];];
     numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
     ];
];
While[Length[eqns]<Plus@@numParamsInPn,
	For[i=1, i<=Length[numParamsInPn],i++,
	If[numParamsInPn[[i]]<=0, Continue[];];
	numParamsInPn[[i]]--;
	If[Min[Transpose[conf . numParamsInPn]]==0,
	(*Not at least one parameter in each equation*)
	numParamsInPn[[i]]++;
	,
	Break[];
	];
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
pointsOnSphere=ParallelTable[SamplePointsOnSphere[dimPs[[i]]+1,numPoints (numParamsInPn[[i]]+1)],{i,Length[dimPs]},DistributedContexts->Automatic];

(*Create system of equations and solve it to find points on CY*)
pointsOnCY=ParallelTable[getPointsOnCY[varsUnflat, numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns,precision],{p,numPoints},DistributedContexts->Automatic];
pointsOnCY=Flatten[pointsOnCY,1];
numPtsPerSample=Length[pointsOnCY];
PrintMsg["Number of points on CY from one ambient space intersection: "<>ToString[numPtsPerSample],frontEnd,verbose];
    
(*Now generate as many points as needed*)
numPoints=Ceiling[numPts/numPtsPerSample];
PrintMsg["Now generating "<>ToString[numPts]<>" points...",frontEnd,verbose];

Clear[x,t];
varsUnflat=Table[Subscript[x,i,a],{i,Length[dimPs]},{a,0,dimPs[[i]]}];
params=Table[Join[{1},Table[Subscript[t,j,k],{k,numParamsInPn[[j]]}]],{j,Length[numParamsInPn]}];
pointsOnSphere=ParallelTable[SamplePointsOnSphere[dimPs[[i]]+1,numPoints (numParamsInPn[[i]]+1)],{i,Length[dimPs]},DistributedContexts->Automatic];
    
(*Create system of equations and solve it to find points on CY*)
If[frontEnd,
    (*pointsOnCY=ResourceFunction["MonitorProgress"][ParallelTable[getPointsOnCY[varsUnflat,numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns],{p,numPoints},DistributedContexts->Automatic]];*)
    pointsOnCY={};
    low=1;
    Monitor[
    For[j=1,j<=20,j++,
    pointsOnCY=Join[pointsOnCY,ParallelTable[getPointsOnCY[varsUnflat,numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns],{p,low, Floor[j numPoints/20]},DistributedContexts->Automatic]];
    low +=Floor[numPoints/20];
    ];
    ,Row[{ProgressIndicator[5(j-1),{1,100}],ToString[5 (j-1)]<>"/100"},"   "]
   ];
    ,
    If[numPoints<=20*numPtsPerSample,
    pointsOnCY=ParallelTable[getPointsOnCY[varsUnflat,numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns],{p,numPoints},DistributedContexts->Automatic];
    ,
    (*Partition in order to provide progress feedback (WolframClient Library ignores messages from subkernels spawned from the kernel used in wl.evaluate(). This negatively impacts performance)*)
    pointsOnCY={};
    low=1;
    For[j=1,j<=20,j++,
    PrintMsg["Generated "<>ToString[5 (j-1)]<>"% of points",frontEnd,verbose];
    pointsOnCY=Join[pointsOnCY,ParallelTable[getPointsOnCY[varsUnflat,numParamsInPn,dimPs,params,Table[pointsOnSphere[[i,p+(b-1) numPoints]],{i,Length[pointsOnSphere]},{b,1+numParamsInPn[[i]]}],eqns],{p,low, Floor[j numPoints/20]},DistributedContexts->Automatic]];
    low = Floor[j numPoints/20];
    ];
    ];
];
PrintMsg["done.",frontEnd,verbose];
pointsOnCY=Flatten[pointsOnCY,1];
Return[{pointsOnCY,numParamsInPn}];
)];

GetPointsOnCYToric[dimCY_,CYEqn_,vars_,sections_,patchMasks_,sectionCoords_,sectionMonoms_,GLSMcharges_,precision_]:=Module[{a,i,j,k,l,eq,dimPs,numEqnsInPn,coeffs,newEqn,sectionSol,sectionCoordsFlat,expMatrix,expMatrixPinv,sVals,logS,logX,xVals,toricVarSols,pts,patchConstraints,patchCoordsList,patchMats,patchMatInverses,patchCoords,logLambda,lambdas,scalings,tmpPts,timeoutResult,validIdx},(
dimPs=Table[Length[sections[[i]]],{i,Length[sections]}];
numEqnsInPn=Table[1,{i,Length[dimPs]}];

(*Precompute exponent matrix and its pseudoinverse for log-linear inversion sections->toric vars*)
sectionCoordsFlat=Flatten[sectionCoords];
expMatrix=Flatten[sections,1];
expMatrixPinv=PseudoInverse[expMatrix];

(*Precompute the per-patch charge matrices and their inverses for log-linear GLSM rescaling*)
patchCoordsList=Table[Flatten[Position[patchMasks[[i]],1]],{i,Length[patchMasks]}];
patchMats=Table[Transpose[GLSMcharges[[All,patchCoordsList[[i]]]]],{i,Length[patchMasks]}];
patchMatInverses=Table[
  Quiet[Check[Inverse[patchMats[[i]]],PseudoInverse[patchMats[[i]]]]],
  {i,Length[patchMasks]}
];

(*Build the equation system: CY + non-CI relations + random patch choices + random sections*)
patchConstraints=Table[RandomChoice[sectionCoords[[a]]]==1.,{a,Length[sectionCoords]}];
eq=Join[CYEqn,patchConstraints];

For[i=1,i<=dimCY,i++,
For[j=1,j<=Length[sections],j++,
If[numEqnsInPn[[j]]>=dimPs[[j]],Continue[];];
coeffs=RandomVariate[NormalDistribution[],{Length[sectionMonoms[[j]]],2}];
newEqn=Sum[(coeffs[[k,1]]+I coeffs[[k,2]]) sectionCoords[[j,k]],{k,Length[coeffs]}];
AppendTo[eq,newEqn==0];
numEqnsInPn[[j]]+=1;
Break[];
];
];

(*Solve for the section values, then invert to toric vars via log-linear algebra*)
timeoutResult=TimeConstrained[
  sectionSol=DeleteCases[Quiet[NSolve[eq]],{}];
  toricVarSols=Table[
    Module[{sv,lS,lX},
      sv=sectionCoordsFlat/.sectionSol[[i]];
      If[AnyTrue[sv,(#==0)||(Head[#]===Complex&&Abs[#]==0)&],
        Nothing,
        lS=Log[sv];
        lX=expMatrixPinv . lS;
        Chop[Exp[lX]]
      ]
    ],
    {i,Length[sectionSol]}
  ];
  pts=DeleteCases[toricVarSols,Nothing];
  "success"
,30,$Failed];

If[timeoutResult===$Failed,
  Return[{{},numEqnsInPn-Table[1,{i,Length[numEqnsInPn]}]}];
];

(*Move each point to its canonical patch (largest coordinate = 1) via log-linear GLSM rescaling*)
Do[
Do[
patchCoords=patchCoordsList[[i]];
(*Skip the patch if any coordinate we'd normalize against is zero*)
If[AnyTrue[pts[[l,patchCoords]],#==0&],Continue[];];
logLambda=patchMatInverses[[i]] . (-Log[pts[[l,patchCoords]]]);
scalings=Exp[logLambda . GLSMcharges];
tmpPts=scalings*pts[[l]];
If[Max[Abs[tmpPts]]<=1+10^-8,
pts[[l]]=Chop[tmpPts];
Break[];
];
,{i,1,Length[patchMasks]}];
,{l,1,Length[pts]}];
Return[{pts,numEqnsInPn-Table[1,{i,Length[numEqnsInPn]}]}];
)];

GenerateToricPointsM[numPts_,dimCY_,coefficients_,exponents_,sections_,sectionRelationCoeffs_,sectionRelationExps_,patchMasks_,GLSMcharges_,precision_:10,verbose_:0,frontEnd_:False]:=Module[{vars,CYeqn,i,j,k,sectionCoords,sectionCoordsFlat,expSectionsFlat,nonCIRelations,linEqCoeffs,lineqs,toricToSections,sectionMonoms,numPoints,pointsOnCY,numPtsPerSample,numEqnsInPn},(
vars=Table[Subscript[x,i],{i,Length[sections]+dimCY+1}];

Clear[s];
sectionCoords=Table[s[a-1,i-1],{a,Length[sections]},{i,Length[sections[[a]]]}];
sectionCoordsFlat=Flatten[sectionCoords];

nonCIRelations={};
For[i=1,i<=Length[sectionRelationCoeffs],i++,
AppendTo[nonCIRelations,0==Sum[sectionRelationCoeffs[[i,a]] Product[sectionCoordsFlat[[r]]^sectionRelationExps[[i,a,r]],{r,Length[sectionCoordsFlat]}],{a,Length[sectionRelationCoeffs[[i]]]}]];
];

expSectionsFlat=Flatten[sections,1];
linEqCoeffs=Table[Subscript[a,r],{r,Length[expSectionsFlat]}];
CYeqn=0;
For[i=1,i<=Length[exponents],i++,
lineqs=Table[linEqCoeffs[[r]]>=0,{r,Length[linEqCoeffs]}];
AppendTo[lineqs,exponents[[i]]==Sum[linEqCoeffs[[i]] expSectionsFlat[[i]],{i,Length[linEqCoeffs]}]];
toricToSections=FindInstance[lineqs,linEqCoeffs,Integers,1];
If[Length[toricToSections]==0,
 If[frontEnd,Print["Something is wrong. Cannot express anticanonical section through KC sections."];,ClientLibrary`error["Something is wrong. Cannot express anticanonical section through KC sections."]];
 Return[{},{0,0}]
 ];
CYeqn+=coefficients[[i]] Product[sectionCoordsFlat[[r]]^toricToSections[[1,r,2]],{r,Length[sectionCoordsFlat]}]
];
CYeqn=Join[{CYeqn==0},nonCIRelations];

sectionMonoms=Table[Table[Times@@(vars^sections[[i,j]]),{j,Length[sections[[i]]]}],{i,Length[sections]}];

(*Distribute the helper once so parallel calls do not re-serialize the context*)
DistributeDefinitions[GetPointsOnCYToric];
ParallelEvaluate[
  Off[ParallelDo::subpar, ParallelTable::subpar, 
      ParallelMap::subpar, ParallelCombine::subpar, ParallelSum::subpar]
];

{pointsOnCY,numEqnsInPn}=GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision];

pointsOnCY=ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,10},DistributedContexts->Automatic];
If[Length[DeleteCases[Table[Length[pointsOnCY[[i]]],{i,Length[pointsOnCY]}],0]]==0,
  PrintMsg["All trial runs returned 0 points; aborting.", frontEnd, verbose];
  Return[{{}, numEqnsInPn}]
];
numPtsPerSample=Min[DeleteCases[Table[Length[pointsOnCY[[i]]],{i,Length[pointsOnCY]}],0]];
PrintMsg["Number of points on CY from one ambient space intersection: "<>ToString[numPtsPerSample],frontEnd,verbose];
pointsOnCY=Flatten[pointsOnCY,1];
numPoints=Ceiling[numPts/numPtsPerSample];
PrintMsg["Now generating "<>ToString[numPts]<>" points...",frontEnd,verbose];

If[frontEnd,
    pointsOnCY={};
    Monitor[
    For[j=1,j<=20,j++,
    pointsOnCY=Join[pointsOnCY,ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,Ceiling[numPoints/20]},DistributedContexts->Automatic]];
    ];
    ,Row[{ProgressIndicator[5(j-1),{1,100}],ToString[5(j-1)]<>"/100"},"   "]
   ];
    ,
    If[numPoints<=20*numPtsPerSample,
    pointsOnCY=ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,numPoints},DistributedContexts->Automatic];
    ,
    pointsOnCY={};
    For[j=1,j<=20,j++,
    PrintMsg["Generated "<>ToString[5(j-1)]<>"% of points",frontEnd,verbose];
    pointsOnCY=Join[pointsOnCY,ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,Ceiling[numPoints/20]},DistributedContexts->Automatic]];
    ];
    ];
];
PrintMsg["done.",frontEnd,verbose];
pointsOnCY=Flatten[pointsOnCY,1];
If[Length[pointsOnCY]>numPts,pointsOnCY=pointsOnCY[[1;;numPts]]];
Return[{pointsOnCY,numEqnsInPn}];
)];
