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
    If[verbose==0||numPoints<=20*numPtsPerSample,
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

GetPointsOnCYToric[dimCY_,CYEqn_,vars_,sections_,patchMasks_,sectionCoords_,sectionMonoms_,GLSMcharges_,precision_]:=Module[{a,i,j,k,l,CYEqnInSections,eq,dimPs,numEqnsInPn,coeffs,newEqn,sectionSol,toricSol,sectionsToToric,toricVarSols,toricVars,pts,patchCoords,scalings,scaleSol,tmpPts,lambdas},( 
dimPs=Table[Length[sections[[i]]],{i,Length[sections]}];(*Length[sections]=h^11 and Length[sections[[i]]]=Number of sections of the i^th KC generator*)
numEqnsInPn=Table[1,{i,Length[dimPs]}]; (*We initialize with 1 since we include one patch constraint s[a,i]==1 per P^n factor*)
(*This contains the toric hypersurface equation and the non-complete intersection relations*)
eq=CYEqn;
(* Choose a random patch, i.e. set a random s[a,i]-> 1 in each P^n *)
For[a=1, a<=Length[sectionCoords],a++,
AppendTo[eq,RandomChoice[sectionCoords[[a]]]==1.];
];

(*Now add random sections*)
(*Need to get points upon intersection with equations i.e. we need to find the right number of sections. We need dimCY many conditions *)
For[i=1,i<=dimCY,i++,
(*Search through sections and take the first one subject to the constraint that we do not overshoot the dimensionality*)
For[j=1,j<=Length[sections],j++,
(*already enough equations in this P^n*)
If[numEqnsInPn[[j]] >= dimPs[[j]],Continue[];];
coeffs=RandomVariate[NormalDistribution[],{Length[sectionMonoms[[j]]],2}];
newEqn=Sum[(coeffs[[k,1]]+I coeffs[[k,2]]) sectionCoords[[j,k]] ,{k,Length[coeffs]}];
AppendTo[eq,newEqn==0];
numEqnsInPn[[j]]+=1;
Break[];
];
];
(*Solve the equations in terms of the s[a,i]*)
sectionSol=DeleteCases[Quiet[NSolve[eq]],{}];
sectionsToToric=Flatten[Table[sectionCoords[[a,i]]->sectionMonoms[[a,i]],{a,Length[sectionCoords]},{i,Length[sectionCoords[[a]]]}]];
(*Find solution in terms of the toric variables*)
Clear[x];
toricVars=vars;
toricVarSols=DeleteCases[Table[FindInstance[(sectionSol[[i]]/. Rule->Equal)/.sectionsToToric,toricVars,Complexes,1]//Chop,{i,Length[sectionSol]}],{}];
pts=Flatten[toricVars/.toricVarSols,1];

(*pts=Quiet[vars/.NSolve[eq,vars,WorkingPrecision->precision]];*)
Clear[\[Lambda]];
lambdas=Table[\[Lambda][k],{k,Length[GLSMcharges]}];
(*go to patch where largest coordinate is 1*)
ParallelDo[
Do[
patchCoords=Flatten[Position[patchMasks[[i]],1]];
scalings=Times@@Power[lambdas,GLSMcharges];
eq=Table[1==pts[[l,patchCoords[[j]]]]*scalings[[patchCoords[[j]]]],{j,Length[patchCoords]}];
scaleSol=Quiet[Solve[eq]];
If[Length[scaleSol]>0,
tmpPts=(scalings*pts[[l]])/.scaleSol[[1]];
];
If[Max[Abs[tmpPts]]==1,
pts[[l]]=Chop[tmpPts];
Break[];
];
,{i,1,Length[patchMasks]}];
,{l,1,Length[pts]}];
Return[{pts,numEqnsInPn-Table[1,{i,Length[numEqnsInPn]}]}];
)];

GenerateToricPointsM[numPts_,dimCY_,coefficients_,exponents_,sections_,sectionRelationCoeffs_,sectionRelationExps_,patchMasks_,GLSMcharges_,precision_:20,verbose_:0,frontEnd_:False]:= Module[{vars,CYeqn,i,j,k,sectionCoords,sectionCoordsFlat,expSectionsFlat,nonCIRelations,secRel,linEqCoeffs,lineqs,toricToSections,sectionMonoms,numPoints,params,allEqns,pointsOnCY,newPoints,numPtsPerSample,numEqnsInPn},( 
vars=Table[Subscript[x,i],{i,Length[sections]+dimCY+1}];

(*Construct the CY equation in terms of the sections*)
Clear[s];
sectionCoords=Table[s[a-1,i-1],{a,Length[sections]},{i,Length[sections[[a]]]}];
sectionCoordsFlat=Flatten[sectionCoords];
(*Reconstruct the non-complete intersection relations*)
nonCIRelations={};
For[i=1,i<=Length[sectionRelationCoeffs],i++,
AppendTo[nonCIRelations,0==Sum[sectionRelationCoeffs[[i,a]] Product[sectionCoordsFlat[[r]]^sectionRelationExps[[i,a,r]],{r,Length[sectionCoordsFlat]}],{a,Length[sectionRelationCoeffs[[i]]]}]];
];

expSectionsFlat=Flatten[sections,1];
linEqCoeffs=Table[Subscript[a, r],{r,Length[expSectionsFlat]}];
CYeqn=0;
For[i=1,i<=Length[exponents],i++,
lineqs=Table[linEqCoeffs[[r]]>=0,{r,Length[linEqCoeffs]}];
AppendTo[lineqs,exponents[[i]]==Sum[linEqCoeffs[[i]] expSectionsFlat[[i]] ,{i,Length[linEqCoeffs]}]];
toricToSections=FindInstance[lineqs,linEqCoeffs,Integers,1];(*There can be more than one way of expressing a monomial in terms of the sections. However, the different cases are captured by the non-CI relations*)
If[Length[toricToSections]==0,
 If[frontEnd,Print["Something is wrong. Cannot express anticanonical section through KC sections."];,ClientLibrary`error["Something is wrong. Cannot express anticanonical section through KC sections."]];
 Return[{},{0,0}](*Should never happen*)
 ];
(*Add the expression to the CY*)
CYeqn+=coefficients[[i]] Product[sectionCoordsFlat[[r]]^toricToSections[[1,r,2]],{r,Length[sectionCoordsFlat]}]
];
CYeqn=Join[{CYeqn==0},nonCIRelations];

(*Reconstruct sections*)
sectionMonoms=Table[Table[Times@@(vars^sections[[i,j]]),{j,Length[sections[[i]]]}],{i,Length[sections]}];
(*Get distribution of parameters*)
{pointsOnCY,numEqnsInPn}=GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision];
(*Generate points on CY.Do 10 trial run to find how many points you get from one intersection*)
pointsOnCY=ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,10},DistributedContexts->Automatic];
numPtsPerSample=Min[Table[Length[pointsOnCY[[i]]],{i,Length[pointsOnCY]}]];
pointsOnCY=Flatten[pointsOnCY,1];
PrintMsg["Number of points on CY from one ambient space intersection: "<>ToString[numPtsPerSample],frontEnd,verbose];

(*Now generate as many points as needed*)
numPoints=Ceiling[(numPts-Length[pointsOnCY])/numPtsPerSample];
PrintMsg["Now generating "<>ToString[numPts]<>" points...",frontEnd,verbose];

(*Create system of equations and solve it to find points on CY*)
If[frontEnd,
    newPoints=ResourceFunction["MonitorProgress"][ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,numPoints},DistributedContexts->Automatic]];
    ,
    If[verbose==0,
    newPoints=ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,numPoints},DistributedContexts->Automatic];
    ,
    (*Partition in order to provide progress feedback (WolframClient Library ignores messages from subkernels spawned from the kernel used in wl.evaluate(). This negatively impacts performance)*)
    newPoints={};
    For[i=1,i<=20,i++,
    PrintMsg["Generated "<>ToString[5 (i-1)]<>"% of points",frontEnd,verbose];
    newPoints=Join[newPoints,ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,Ceiling[numPoints/20]},DistributedContexts->Automatic]];
    ];
    ];
];
pointsOnCY=Join[pointsOnCY,Flatten[newPoints,1]];
If[Length[pointsOnCY]>numPts,pointsOnCY=pointsOnCY[[1;;numPts]]];
PrintMsg["done.",frontEnd,verbose];
(*PrintMsg["Section distribution: "<>ToString[numEqnsInPn],frontEnd,verbose];*)
Return[{pointsOnCY,numEqnsInPn}];
)];
