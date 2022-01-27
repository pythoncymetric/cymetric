(* ::Package:: *)

(* ::Input::Initialization:: *)
BeginPackage["cymetric`"]


(* ::Input::Initialization:: *)
(*
This package provides a Mathematica interface for the Python cymetric paper. Authors Fabian Ruehle and Robin Schneider. For more information visit https://github.com/pythoncymetric/cymetric 
*)
GetSetting::usage="GetSetting[k] retrieves the value of k from $SETTINGSFILE.\n* Input:\n  - k (string): The entry whose value shall be retrieved\n* Return:\n  - The value v (string) of entry k, or Null if entry k not found\n* Example:\n  - GetSetting[\"Python\"]";
ChangeSetting::usage="ChangeSetting[k, v] sets the value of entry k to v in $SETTINGSFILE.\n* Input:\n  - k (string): The entry whose value shall be set\n  - v (string): The value for entry k\n* Return:\n  - Null\n* Example:\n  - ChangeSetting[\"Python\",\"/usr/bin/python3\"]";
DeleteSetting::usage="DeleteSetting[k] deletes the entry k in $SETTINGSFILE.\n* Input:\n  - k (string): The entry that shall be deleted\n* Return:\n  - Null\n* Example:\n  - DeleteSetting[\"Python\"]";
Setup::usage="Setup[path, Options] finds a valid Python executable; if setup has been run before, it uses the path from global settings. Else it finds a good Python 3 interpreter and creates a virtual environment and patches mathematica to work with python >= 3.7.\n* Input:\n  - path (string): Sets up a python venv in path (defaults to ./venv).\n* Options (run Options[Setup] to see default values):\n  - ForceReinstall (bool): Whether the venv should be reinstalled if it already exists\n* Return:\n  - python (string): path to python venv executable\n* Example:\n  - Setup[]";
GeneratePoints::usage="GeneratePoints[poly,dimPs,variables,Options] generates points on the CY specified by poly in an ambient space of dimPs.\n* Input:\n  - poly (list of polynomials): list of polynomials that define the CY.\n  - dimPs (list of ints): list of dimensions of the product of projective ambient spaces.\n  - variables [optional] (list of list of vars): list of list of variables (one for each projective ambient space). If not provided, alphabetical ordering is assumed.\n* Options (run Options[GeneratePoints] to see default values):\n  - KahlerModuli (list of floats): Kahler moduli \!\(\*SubscriptBox[\(t\), \(i\)]\) of the i'th ambient space factor (defaults to all 1)\n  - Points (int): Number of points to generate (defaults to 200,000)\n  - Precision (int): WorkingPrecision to use when generating the points (defaults to 20)\n  - VolJNorm (float): Normalization for the volume, input int_X J^n at t1=t2=...=1 (defaults to 1)\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Dir (string): Directory where points will be saved\n  - Verbose (int): Verbose level (the higher, the more info is printed)\n* Return:\n  - res (object): Null if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - GeneratePoints[{\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(0\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(1\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(2\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(3\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(4\)], \(5\)]\)},{4},\"Points\"\[Rule]200000,\"KahlerModuli\"\[Rule]{1},\"Dir\"\[Rule]\"./test\"]";
GeneratePointsToric::usage="GeneratePointsToric[PathToToricInfo,Options] generates points on the CY specified by the toric variety that has been analyzed by SAGE in PathToToricInfo. \n* Input:\n  - PathToToricInfo (str): Path where SAGE stored the info on the toirc CY.\n* Options (run Options[GeneratePoints] to see default values):\n  - KahlerModuli (list of floats): Kahler moduli \!\(\*SubscriptBox[\(t\), \(i\)]\) of the i'th ambient space factor (defaults to all 1)\n  - Points (int): Number of points to generate (defaults to 200,000)\n  - Precision (int): WorkingPrecision to use when generating the points (defaults to 20)\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Dir (string): Directory where points will be saved\n  - Verbose (int): Verbose level (the higher, the more info is printed)\n* Return:\n  - res (object): Null if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - GeneratePoints[{\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(0\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(1\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(2\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(3\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(4\)], \(5\)]\)},{4},\"Points\"\[Rule]200000,\"KahlerModuli\"\[Rule]{1},\"Dir\"\[Rule]\"./test\"]";
TrainNN::usage="TrainNN[Options] trains a NN to approximate the CY metric.\n* Options (run Options[TrainNN] to see default values):\n  - Model (string): Choices are:\n     - PhiFS: The NN learns a scalar function \[Phi] s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)+\[PartialD]\!\(\*OverscriptBox[\(\[PartialD]\), \(_\)]\)\[Phi]\n     - PhiFSToric: Like PhiFS, but for CYs built from 4D toric varieties (need to specify location of toric info as generated from SAGE in Option ToricDataPath).\n     - MultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) * (id + \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and \"*\" is component-wise multiplication \n     - MatrixMultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)(id + \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and the multiplication is standard matrix multiplication \n     - AddFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) + \!\(\*SubscriptBox[\(g\), \(NN\)]\) \n     - Free: The NN learns the CY metric directly, i.e. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(NN\)]\)  \n     - MatrixMultFSToric: Like MatrixMultFS, but for CYs built from 4D toric varieties (need to specify location of toric info as generated from SAGE in Option ToricDataPath).\n  - EvaluateModel (bool): If True, computes different quantities for the test set, such as the \[Sigma]-measure, Kahler Loss, Transition Loss, change of Kahler class, and Ricci Scalar. Especially the Kahler and Ricci evaluations can be very slow, so only activate if feasible performance-wise  \n  - HiddenLayers (list of ints): Number of hidden nodes in each hidden layer \n  - ActivationFunctions (list of strings): Tensorflow activation function to use\n  - Epochs (int): Number of training epochs\n  - BatchSize (int): Batch size for training\n  - Kappa (float): Value for kappa=\[Integral]|\[CapitalOmega]|^2/\[Integral] J^3; if 0 is passed, it will be computed automatically.\n  - Alphas (list of floats): Relative weight of losses (Sigma, Kahler, Transition, Ricci, volK); at the moment, the value for Ricci is ignored since we do not train against the Ricci loss\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Dir (string): Directory where the training points will be read from and the trained NN will be saved to\n  - Verbose (int): Verbose level (the higher, the more info is printed)\n* Return:\n  - res (object): Null if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - TrainNN[\"HiddenLayers\"\[Rule]{64,64,64},\"ActivationFunctions\"\[Rule]{\"gelu\",\"gelu\",\"gelu\"},\"Epochs\"\[Rule]20,\"BatchSize\"\[Rule]64,\"Dir\"\[Rule]\"./test\"]";
GetPoints::usage="GetPoints[dataset,Options] gets the CY points generated by the point generator.\n* Input:\n  - dataset (string): \"train\" for training dataset, \"val\" for validation dataset, \"all\" for both training and validation dataset\n* Options (run Options[GetPoints] to see default values):\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Dir (string): Directory where points are saved\n* Return:\n  - res (object): points (as list of complex lists) if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - GetPoints[\"val\",\"Dir\"\[Rule]\"./test\"]";
GetFSWeights::usage="GetFSWeights[dataset,Options] gets the weights of the CY points generated by the point generator for the Fubini-Study metric.\n* Input:\n  - dataset (string or list): \"train\" for training dataset, \"val\" for validation dataset, \"all\" for both training and validation dataset, or a list of points\n* Options (run Options[GetFSWeights] to see default values):\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Dir (string): Directory where weights are saved\n  - Dir (string): Directory where CY information is saved\n* Return:\n  - res (object): weights (as list of floats) if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - GetFSWeights[\"val\",\"Dir\"\[Rule]\"./test\"]";
GetOmegaSquared::usage="GetOmegaSquared[dataset,Options] gets (|\[CapitalOmega]|\!\(\*SuperscriptBox[\()\), \(2\)]\) of the CY generated by the point generator.\n* Input:\n  - dataset (string): \"train\" for training dataset, \"val\" for validation dataset, \"all\" for both training and validation dataset, or a list of points\n* Options (run Options[GetOmegaSquared] to see default values):\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Dir (string): Directory where Omegas are saved\n* Return:\n  - res (object): (|\[CapitalOmega]|\!\(\*SuperscriptBox[\()\), \(2\)]\) (as list of floats) if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - GetOmegaSquared[\"val\",\"Dir\"\[Rule]\"./test\"]";
GetCYWeights::usage="GetCYWeights[dataset,Options] gets the weights of the CY points for the CY metric.\n* Input:\n  - dataset (string): \"train\" for training dataset, \"val\" for validation dataset, \"all\" for both training and validation dataset, or a list of points\n* Options (run Options[GetCYWeights] to see default values):\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Model (string): Choices are:\n     - PhiFS: The NN learns a scalar function \[Phi] s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)+\[PartialD]\!\(\*OverscriptBox[\(\[PartialD]\), \(_\)]\)\[Phi]\n     - MultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) * (id + \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and \"*\" is component-wise multiplication \n     - MatrixMultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)(id + \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and the multiplication is standard matrix multiplication \n     - AddFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) + \!\(\*SubscriptBox[\(g\), \(NN\)]\) \n     - Free: The NN learns the CY metric directly, i.e. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(NN\)]\)\n  - Kappa (float): Value for kappa=\[Integral]|\[CapitalOmega]|^2/\[Integral] J^3; if 0 is passed, it will be computed automatically.\n  - Dir (string): Directory where weights and the trained NN are saved\n* Return:\n  - res (object): weights (as list of floats) if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - GetCYWeights[\"val\",\"Dir\"\[Rule]\"./test\"]";
GetPullbacks::usage="GetPullbacks[points,Options] computes the pullbacks from the ambient space coordinates to the CY at the given points. You need to call GeneratePoints[] first. The pullback will work for any point (not just the generated ones), but GeneratePoints[] computes several quantities (such as derivatives) needed for the pullback matrix. \n* Input:\n  - points (list): list of list of complex numbers that specify points on the CY\n* Options (run Options[GetPullbacks] to see default values):\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Dir (string): Directory where the point generator is saved.\n* Return:\n  - res (object): pullbacks (as list of float matrices) if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the computation (for potential faster future reuse) \n* Example:\n  - GetPullbacks[{{1,2,3,4,5}},\"Dir\"\[Rule]\"./test\"]";
CYMetric::usage="CYMetric[points,Options] computes the CY metric at the given points.\n* Input:\n  - points (list): list of list of complex numbers that specify points on the CY\n* Options (run Options[CYMetric] to see default values):\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - Model (string): Choices are:\n     - PhiFS: The NN learns a scalar function \[Phi] s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)+\[PartialD]\!\(\*OverscriptBox[\(\[PartialD]\), \(_\)]\)\[Phi]\n     - MultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) * (id + \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and \"*\" is component-wise multiplication \n     - MatrixMultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)(id + \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and the multiplication is standard matrix multiplication \n     - AddFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) + \!\(\*SubscriptBox[\(g\), \(NN\)]\) \n     - Free: The NN learns the CY metric directly, i.e. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(NN\)]\)\n  - Kappa (float): Value for kappa=\[Integral]|\[CapitalOmega]|^2/\[Integral] J^3; if 0 is passed, it will be computed automatically.\n  - Dir (string): Directory where weights and the trained NN are saved\n* Return:\n  - res (object): CY metrics (as list of float matrices) if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the coomputation (for potential faster future reuse) \n* Example:\n  - CYMetric[{{1,2,3,4,5}},\"Dir\"\[Rule]\"./test\"]";
FSMetric::usage="FSMetric[points,Options] computes the pullback of the Fubini-Study metric from the ambient space to the CY at the given points.\n* Input:\n  - points (list): list of list of complex numbers that specify points on the CY\n* Options (run Options[FSMetric] to see default values):\n  - Python (string): python executable to use (defaults to the one of $SETTINGSFILE)\n  - Session (ExternalSessionObject): Python session with all dependencies loaded (if not provided, a new one will be generated)\n  - KahlerModuli(list): Point in Kahler moduli space.\n  - Dir (string): Directory where the point generator is saved.\n* Return:\n  - res (object): FS metrics at input points (as list of float matrices) if no error occured, otherwise a string with the error\n  - session (ExternalSessionObject): The session object used in the computation (for potential faster future reuse) \n* Example:\n  - FSMetric[{{1,2,3,4,5}},\"Dir\"\[Rule]\"./test\"]";
DiscoverPython::usage="DiscoverPython[useForVENV,Options] finds a Python executable for python3 with version <3.9.\n* Input:\n  - useForVENV (string): If False, keeps looking until it finds an environment with all necessary packages installed. Otherwise just uses the latest Python 3 version <3.9\n* Options (run Options[DiscoverPython] to see default values):\n  - Version (string): Specify a Python version to use \n* Return:\n  - python (string): path to python venv executable\n* Example:\n  - DiscoverPython[]";
SetupPythonVENV::usage="SetupPythonVENV[exec,Options] sets up a Python virtual environment.\n* Input:\n  - exec (string): Path to Python that will be used to create the venv.\n* Options (run Options[SetupPythonVENV] to see default values):\n  - VENVPath (string): Path where venv will be created\n  - Patch (bool): Whether Mathematica should be patched to fix a bug for Python >3.5 (this only changes a text file used for Python interop and does not interfere with the Mathematica installation) \n* Return:\n  - python (string): path to python venv executable\n* Example:\n  - SetupPythonVENV[\"/usr/bin/python3\"]";
GetSession::usage="GetSession[exec,mysession] returns a Python session with the cymetric package loaded.\n* Input:\n  - exec (string): Path to Python that will be used to create the session. If not specified, the installation from $SETTINGSFILE is used\n  - mysession (ExternalSessionObject): If mysession is a valid Python session, the cypackage and all dependencies are loaded intyo the session if necessary \n* Return:\n  - session (ExternalSessionObject): session with cymetric package and all dependencies loaded\n* Example:\n  - GetSession[]";



(* ::Input::Initialization:: *)
Begin["`Private`"];
$SETTINGSFILE=Quiet[Check[FileNameJoin[{NotebookDirectory[],FileBaseName[NotebookFileName[]]<>"-cymetricsettings.txt"}],""]];
$WORKDIR=Quiet[Check[NotebookDirectory[],"~/"]];
GetSetting::fileNotFound="Could not retrieve `1`. No settings file found.";
GetSetting[k_]:=Module[{settings},(
If[!FileExistsQ[$SETTINGSFILE],Message[GetSetting::fileNotFound,k];Return[];];
settings=ToExpression[Import[$SETTINGSFILE]];
If[KeyExistsQ[settings,k],Return[settings[k]];];
Return[Null];
)];

ChangeSetting[k_,v_]:=Module[{settings,hnd},(
If[!FileExistsQ[$SETTINGSFILE],
Export[$SETTINGSFILE,Association[]];
];
settings=ToExpression[Import[$SETTINGSFILE]];
settings[k]=v;
hnd=OpenWrite[$SETTINGSFILE];
Write[hnd,settings];
Close[hnd];
Return[];
)];

DeleteSetting[k_]:=Module[{settings,hnd},(
If[!FileExistsQ[$SETTINGSFILE],
Export[$SETTINGSFILE,Association[]];
];
settings=ToExpression[Import[$SETTINGSFILE]];
If[KeyExistsQ[settings,k],KeyDropFrom[settings,k]];
hnd=OpenWrite[$SETTINGSFILE];
Write[hnd,settings];
Close[hnd];
Return[];
)];
ToPython::usage="ToPython[input] returns input in Python format.\n* Input:\n  - input (string or list): Mathematica expression\n* Return:\n  - input converted to Python syntax\n* Example:\n  - ToPython[{1.3 \[ImaginaryI]}]";
ToPython[input_]:=(
SetOptions[$Output,PageWidth->Infinity];
If[StringQ[input],Return[input]];
Return[StringReplace[ToString[InputForm[input,NumberMarks->False]], {"{"->"[","}"->"]", "*^"->"e", "^"->"**", "I"->"1.j"}]];
)

Options[Setup]={"ForceReinstall"->False};
Setup[path_String:FileNameJoin[{$WORKDIR,"venv"}],OptionsPattern[]]:=Module[{exec,res,settings,forceReinstall,python, packageDir,session},(
forceReinstall=OptionValue["ForceReinstall"];
(*Auto-detect whether venv is already set up in the folder*)
If[!forceReinstall,
python=Quiet[Check[GetSetting["Python"],Null]];
If[!python===Null,
If[!FileExistsQ[python],Print["Found settings file for this notebook, but the path to the virtual environment specified in this file does not exist. Please set a new one with changeSettings[Python-><path/to/python>], or run again with option ForceReinstall->True"];Return[""];
];
session=StartExternalSession[<|"System"->"Python","Executable"->python|>];
packageDir=ExternalEvaluate[session,"import cymetric;import os;os.path.dirname(cymetric.__file__)"];
Begin["cymetric`Private`"];
Get[FileNameJoin[{packageDir,"wolfram/PointGeneratorMathematica.m"}]];
End[];
Return[python];];
If[FileExistsQ[$SETTINGSFILE],Print["Settings file does not contain a path to a Python environment. Please set one with changeSettings[Python-><path/to/python>], or run again with option ForceReinstall->True"];Return[""];
];
];
exec=DiscoverPython[True];
res=SetupPythonVENV[exec,"Patch"->True,"VENVPath"->path];
ChangeSetting["Python",res];
session=StartExternalSession[<|"System"->"Python","Executable"->res|>];
packageDir=ExternalEvaluate[session,"import cymetric;import os;os.path.dirname(cymetric.__file__)"];
(*Import the mathematica point generation functions into the current session*)
Begin["cymetric`Private`"];
Get[FileNameJoin[{packageDir,"wolfram/PointGeneratorMathematica.m"}]];
End[];
Return[res];
)];

Options[DiscoverPython]={"Version"->Null};
DiscoverPython[useForVENV_:False,OptionsPattern[]]:=Module[{pythonEnvs,exec,session,res,version,i},(
version=OptionValue["Version"];
pythonEnvs=FindExternalEvaluators["Python","ResetCache"->True];
pythonEnvs=Reverse[SortBy[pythonEnvs,"Version"]];(*Start with latest Python on tyhe system and work your way down. ATM, there's no TF for python 3.9, so we skip that*)
Print["Mathematica discovered the following Python environments on your system:"];
Print[pythonEnvs];
Print["Looking for Python 3"];
For [i=1,i<=Length[pythonEnvs],i++,
If[StringTake[pythonEnvs[i]["Version"],1]!="3"||StringTake[pythonEnvs[i]["Version"],3]=="3.9", Continue[]];
If[!version===Null,If[version!=pythonEnvs[i]["Version"],Continue[];]];
exec=pythonEnvs[i]["Executable"];
Print["Found Python version ",pythonEnvs[i]["Version"], " at ",exec,"."];
If[useForVENV,Return[exec]];
Print["Looking for pyzmq for Mathematica interop and tensorflow..."];
session=Quiet[Check[StartExternalSession[<|"System"->"Python","Executable"->exec|>],Print["Mathematica couldn't start this Python session. Trying next Python environment..."];Continue[]]];
DeleteObject[session];
(*Attempt to load*)
session=GetSession[exec];
If[session===Null,Continue[];,Print["Found good Python environment: ",exec]];
Quiet[Check[DeleteObject[session],Continue[]]];
Return[exec];
];
)];

Options[SetupPythonVENV]={"VENVPath"->FileNameJoin[{$WORKDIR,"venv"}],"Patch"->False};
SetupPythonVENV[exec_String,OptionsPattern[]]:=Module[{path,patchEE,setupVENV,python,pip,installPackages,packages,session,res,hnd,content,i},(
path=OptionValue["VENVPath"];
patchEE=OptionValue["Patch"];
Print["Creating virtual environment at ",path];
setupVENV=RunProcess[{exec,"-m","venv",path}];
If[setupVENV["ExitCode"]!=0,Print["An error occurred. Here's the output"];Print[setupVENV["StandardOutput"]];Print[setupVENV["StandardError"]];Return[];];
python=FileNameJoin[{path,"bin","python"}];
pip=FileNameJoin[{path,"bin","pip"}];
If[!FileExistsQ[pip],pip=FileNameJoin[{path,"bin","pip3"}];];
If[!FileExistsQ[pip],Print["Error: Couldn't find pip at ",FileNameJoin[{path,"bin"}]];Return[python];];
(*Install packages*);
packages={ "h5py","joblib","numpy","pyyaml","pyzmq","scipy","sympy","wolframclient"};
Print["Upgrading pip..."];
installPackages=RunProcess[{pip,"install", "--upgrade", "pip"}];
If[installPackages["ExitCode"]!=0,Print["An error occurred. Here's the output"];Print[installPackages["StandardOutput"]];Print[installPackages["StandardError"]];Return[python];];
For[i=1,i<=Length[packages],i++,
Print["Installing ",packages[[i]],"..."];
installPackages=RunProcess[{pip,"install",packages[[i]]}];
If[installPackages["ExitCode"]!=0,Print["An error occurred. Here's the output"];Print[installPackages["StandardOutput"]];Print[installPackages["StandardError"]];Return[python];
];
];
(*Install CY package*)
Print["Installing cymetric..."];
installPackages=RunProcess[{pip,"install","git+https://github.com/pythoncymetric/cymetric.git"}];
If[installPackages["ExitCode"]!=0,Print["An error occurred. Here's the output"];Print[installPackages["StandardOutput"]];Print[installPackages["StandardError"]];Return[python];
];
(*Register for use with Mathematica*)
Print["Registering venv with mathematica..."];
RegisterExternalEvaluator["Python",python];
Print["Checking whether 'externalevaluate.py' needs to be patched for this version..."];
session=StartExternalSession[<|"System"->"Python","Executable"->python|>];
res=Quiet[ExternalEvaluate[session,{"import os;x=0;x"}]];
If [ListQ[res],res= res[[1]]];
If[res["Message"]=="required field \"type_ignores\" missing from Module",
Print["Patch needs to be applied"];
If[!patchEE,
Print["Option for automatically applying patch not set. Please apply manually and try again."];
DeleteObject[session];
Return[python];
,
Print["Applying patch..."];
hnd=OpenRead[FileNameJoin[{ $InstallationDirectory,"/SystemFiles/Links/WolframClientForPython/wolframclient/utils/externalevaluate.py"}]];
content=ReadString[hnd];
Close[hnd];
content=StringReplace[content,{"exec(compile(ast.Module(expressions), '', 'exec'), current)"->"exec(compile(ast.Module(expressions, []), '', 'exec'), current)  # exec(compile(ast.Module(expressions), '', 'exec'), current) # changed by CYMetrics"}];
hnd=OpenWrite[FileNameJoin[{ $InstallationDirectory,"/SystemFiles/Links/WolframClientForPython/wolframclient/utils/externalevaluate.py"}]];
WriteString[hnd,content];
Close[hnd];
];
];
Print["Testing new environment..."];
session=GetSession[python];
If[!session===Null,DeleteObject[session],Return[python]];
Print["Everything is working!"];
Return[python];
)];

GetSession[exec_:Null,mysession_:Null]:=
Module[{validSession,session,python,res},(
python=exec;
session=mysession;
validSession=Quiet[Check[ExternalEvaluate[session,"2+2==4"],False]];
If[session===Null||!validSession,
If[python===Null,python=Check[GetSetting["Python"],Null]];
If[python===Null,Print["No python interpreter configured. Please provide one with the option \"Python-><path/to/python>\""];Return[Null];];
(*Start session, load in prepare data file*)
session=StartExternalSession[<|"System"->"Python","Executable"->python|>];
res=ExternalEvaluate[session,"import cymetric;import cymetric.wolfram.mathematicalib as mcy;"];
If[FailureQ[res], Print[res];Return[Null];];
];
(*Check whether mathematica_lib.py has been loaded; if not, load it into session*)
res=ExternalEvaluate[session,"import sys;'cymetric' in sys.modules"];
If[FailureQ[res], Print[res];Return[Null];];
If[!res,res=ExternalEvaluate[session,"import cymetric;import cymetric.wolfram.mathematicalib as mcy;"]];
If[FailureQ[res], Print[res];Return[Null];];
Return[session];
)];

Options[GeneratePoints]={"KahlerModuli"->{},"Points"->200000,"Precision"->20,"VolJNorm"->1,"Python"->Null,"Session"->Null,"Dir"->FileNameJoin[{$WORKDIR,"test"}],"Verbose"->3};
GeneratePoints[poly_,dimPs_,variables_List:{},OptionsPattern[]]:=
Module[{python,points,numPts,pointsFile,verbose,session,outDir,res,startPos, monomials, coeffs,kahlerModuli, precision,loggerLevel,args,vars,randomPoint,i,prev,curr,isSymmetric,pointsBatched,volJNorm,numParamsInPn},(
python=OptionValue["Python"];
numPts=OptionValue["Points"];
verbose=OptionValue["Verbose"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
kahlerModuli=OptionValue["KahlerModuli"];
precision=OptionValue["Precision"];
volJNorm=OptionValue["VolJNorm"];
(*Start session, load in prepare data file*)
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];

loggerLevel="ERROR";(*Taken for verbose \[LessEqual] 0*)
If[verbose==1, loggerLevel="WARNING"];
If[verbose==2, loggerLevel="INFO"];
If[verbose>=3, loggerLevel="DEBUG"];

(*Find coefficients and exponents of the polynomials*)
If[!ListQ[poly],poly={poly}];
If[variables=={},
If[verbose >0,Print["Warning: No variables specified, assuming alphabetical monomial ordering."]];
vars=Sort[Variables[Flatten[poly]]];
If[verbose >1,
Print["Variables have been assigned to the ambient space factors as follows:"];
startPos=1;
For[i=1,i<=Length[dimPs],i++,
Print[i,".) ",SymbolName[P]^ToString[dimPs[[i]]],": ",vars[[startPos;;startPos+dimPs[[i]]]]];
startPos+=dimPs[[i]]+1;
];
,
vars=Flatten[vars];
];
,
vars=Flatten[variables];
];
monomials={};coeffs={};
For[i=1,i<=Length[poly],i++,
res=Association[CoefficientRules[poly[[i]],vars]];
AppendTo[monomials,Keys[res]];
AppendTo[coeffs ,Values[res]];
];
(*If[Length[poly]\[Equal]1,monomials=monomials[[1]];coeffs=coeffs[[1]]];*)
(*Return[{monomials,coeffs}];*)
kahlerModuli = If[kahlerModuli=={}, Table[1,{i,Length[dimPs]}],kahlerModuli];
pointsFile=FileNameJoin[{outDir,"points.pickle"}];
Print["Generating ",numPts," points..."];
{points,numParamsInPn}=GeneratePointsM[numPts,dimPs,coeffs,monomials,precision,verbose,True];
Print["Writing points to ",pointsFile];
If[!DirectoryQ[outDir],CreateDirectory[outDir]];
(* batch the conversion, otherwise there is an error for long points*)
pointsBatched=Partition[points,UpTo[1000]];
ExternalEvaluate[session,"generated_pts=[];"];
For[i=1,i<=Length[pointsBatched],i++,
res=ExternalEvaluate[session,"generated_pts +="<>ToPython[pointsBatched[[i]]]<>";"];
If[FailureQ[res],Print["An error occurred."];Print[res];Break[];];
];
res=ExternalEvaluate[session,"import pickle;import numpy as np;pickle.dump(np.array(generated_pts), open(\""<>pointsFile<>"\",'wb'));"];
ExternalEvaluate[session,"generated_pts=[];"];
If[FailureQ[res],Print["An error occurred."];Print[res];];

args="{
        'outdir':        \""<>ToPython[outDir]<>"\",
        'logger_level':  logging."<>loggerLevel<>",
        'num_pts':       "<>ToPython[numPts]<>",
        'monomials':     "<>ToPython[monomials]<>",       
        'coeffs':        "<>ToPython[coeffs]<>",
        'k_moduli':      "<>ToPython[kahlerModuli]<>",
        'ambient_dims':  "<>ToPython[dimPs] <>",
        'precision':     "<>ToPython[precision] <>",
        'vol_j_norm':    "<>ToPython[volJNorm] <>",
        'selected_t': "<>ToPython[numParamsInPn] <>",
        'point_file_path':\""<>ToPython[pointsFile] <>"\"
       }";
res=ExternalEvaluate[session,"mcy.generate_points"->args];
If[FailureQ[res],
Print["An error occurred."];Print[res];,
DeleteFile[pointsFile];
];
Return[{res,session}];
)];

Options[GeneratePointsToric]={"KahlerModuli"->{},"Points"->200000,"Precision"->20,"Python"->Null,"Session"->Null,"Dir"->FileNameJoin[{$WORKDIR,"test"}],"Verbose"->3};
GeneratePointsToric[PathToToricInfo_,OptionsPattern[]]:=
Module[{python,points,numPts,pointsFile,verbose,session,outDir,res,startPos, monomials, coeffs,kahlerModuli, precision,loggerLevel,args,vars,randomPoint,i,prev,curr,isSymmetric,pointsBatched,volJNorm,dimCY,coefficients,dimPs,patchMasks,GLSMcharges,sections},(
python=OptionValue["Python"];
numPts=OptionValue["Points"];
verbose=OptionValue["Verbose"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
kahlerModuli=OptionValue["KahlerModuli"];
precision=OptionValue["Precision"];
(*Start session, load in prepare data file*)
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];

loggerLevel="ERROR";(*Taken for verbose \[LessEqual] 0*)
If[verbose==1, loggerLevel="WARNING"];
If[verbose==2, loggerLevel="INFO"];
If[verbose>=3, loggerLevel="DEBUG"];

(*Read in toric data*)
res=ExternalEvaluate[session,"import pickle;import numpy as np;pickle.load(open(\""<>PathToToricInfo<>"\",'rb'))"];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[res,session]];
dimCY=res[["dim_cy"]];
monomials=res[["exp_aK"]];
coefficients=res[["coeff_aK"]];
patchMasks=res[["patch_masks"]];
GLSMcharges=res[["glsm_charges"]];
sections=res[["exps_sections"]];
volJNorm=res["vol_j_norm"];
dimPs={Length[monomials[[1]]]};
kahlerModuli = If[kahlerModuli=={}, Table[1,{i,Length[dimPs]}],kahlerModuli];
pointsFile=FileNameJoin[{outDir,"points.pickle"}];
args="{
        'toric_file_path': \""<>PathToToricInfo<>"\",
        'outdir':        \""<>ToPython[outDir]<>"\",
        'logger_level':  logging."<>loggerLevel<>",
        'num_pts':       "<>ToPython[numPts]<>",
        'dim_cy':        "<>ToPython[dimCY]<>",
        'monomials':     "<>ToPython[monomials]<>",
        'coeffs':        "<>ToPython[coefficients]<>",
        'k_moduli':      "<>ToPython[kahlerModuli]<>",
        'ambient_dims':  "<>ToPython[dimPs]<>",
        'sections':      "<>ToPython[sections]<>",
        'patch_masks':   "<>ToPython[patchMasks]<>",
        'glsm_charges':  "<>ToPython[GLSMcharges]<>",
        'precision':     "<>ToPython[precision] <>",
        'vol_j_norm':    "<>ToPython[volJNorm] <>",
        'verbose':       "<>ToPython[verbose] <>",
        'point_file_path':\""<>ToPython[pointsFile]<>"\"
       }";
Print["Generating ",numPts," points..."];
points=GenerateToricPointsM[numPts,dimCY,coefficients,monomials,sections,patchMasks,GLSMcharges,precision,verbose];
Print["Writing points to ",pointsFile];
If[!DirectoryQ[outDir],CreateDirectory[outDir]];
(* batch the conversion, otherwise there is an error for long points*)
pointsBatched=Partition[points,UpTo[1000]];
ExternalEvaluate[session,"generated_pts=[];"];
For[i=1,i<=Length[pointsBatched],i++,
res=ExternalEvaluate[session,"generated_pts +="<>ToPython[pointsBatched[[i]]]<>";"];
If[FailureQ[res],Print["An error occurred."];Print[res];Break[];];
];
res=ExternalEvaluate[session,"import pickle;import numpy as np;pickle.dump(np.array(generated_pts), open(\""<>pointsFile<>"\",'wb'));"];
ExternalEvaluate[session,"generated_pts=[];"];
If[FailureQ[res],Print["An error occurred."];Print[res];];
res=ExternalEvaluate[session,"mcy.generate_points_toric"->args];
If[FailureQ[res],
Print["An error occurred."];Print[res];,
DeleteFile[pointsFile];
];
Return[{res,session}];
)];

Options[TrainNN]={"Model"->"MultFS","EvaluateModel"->False,"HiddenLayers"->{64,64,64},"ActivationFunctions"->{"gelu","gelu","gelu"},"Epochs"->20,"BatchSize"->64,"Kappa"->0.,"Alphas"->{1.,1.,1.,1.,1.},"Python"->Null,"Session"->Null,"Dir"->FileNameJoin[{$WORKDIR,"test"}],"ToricDataPath"->"","Verbose"->3};
TrainNN[OptionsPattern[]]:=
Module[{python,model,callbacks,nHiddens,acts,nEpochs,batchSize,kappa,alphas,outDir,verbose,session,res, loggerLevel,args,validSession,toricDataPath},(
python=OptionValue["Python"];
outDir=OptionValue["Dir"];
session=OptionValue["Session"];
model=OptionValue["Model"];
callbacks=OptionValue["EvaluateModel"];
nHiddens=OptionValue["HiddenLayers"];
acts=OptionValue["ActivationFunctions"];
nEpochs=OptionValue["Epochs"];
batchSize=OptionValue["BatchSize"];
kappa=OptionValue["Kappa"];
alphas=OptionValue["Alphas"];
verbose=OptionValue["Verbose"];
toricDataPath=OptionValue["ToricDataPath"];

If[toricDataPath==""&&(model=="PhiFSToric"||model=="MatrixMultFSToric"),Print["Need to specify path to toric info as generated by SAGE."];Return[{"Need to specify path to toric info as generated by SAGE.",session}]];

session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];

loggerLevel="ERROR";(*Taken for verbose \[LessEqual] 0*)
If[verbose==1, loggerLevel="WARNING"];
If[verbose==2, loggerLevel="INFO"];
If[verbose>=3, loggerLevel="DEBUG"];

args="{
        'outdir':        \""<>ToPython[outDir]<>"\",
        'logger_level':  logging."<>loggerLevel<>",
        'model':         \""<>ToPython[model]<>"\",
        'callbacks':     "<>ToPython[callbacks]<>",
        'n_hiddens':     "<>ToPython[nHiddens]<>",
		'acts':          "<>ToPython[acts]<>",
		'n_epochs':      "<>ToPython[nEpochs]<>",
		'batch_size':    "<>ToPython[batchSize]<>",
		'kappa':         "<>ToPython[kappa]<>",
        'alphas':        "<>ToPython[alphas]<>",
        'toric_data_path':\""<>ToPython[toricDataPath]<>"\"
       }";
If[verbose>=3,Print[args]];
res=ExternalEvaluate[session,"mcy.train_NN"->args];
If[FailureQ[res],Print["An error occurred"];Print[res];Return[res,session]];
Print["Writing training information to "<>FileNameJoin[{outDir,"training_history_mathematica.m"}]];
Export[FileNameJoin[{outDir,"trianing_history_mathematica.m"}],res];
Return[{res,session}];
)];

Options[GetPoints]={"Python"->Null,"Session"->Null,"Dir"->FileNameJoin[{$WORKDIR,"test"}]};
GetPoints[dataset_:"all",OptionsPattern[]]:=
Module[{python,session,outDir,res,lenPts},(
python=OptionValue["Python"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];

If[dataset=="all",
res=Join[
ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['X_train']"],ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['X_val']"]
];,
res=ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['X_"<>dataset<>"']"]
];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];

If[Length[res]>0,
lenPts=Floor[Length[res[[1]]]/2];
res=res[[;;,1;;lenPts]]+I*res[[;;,lenPts+1;;]];
,
res={};
];
Return[{Chop[Normal[res]],session}];
)];

Options[GetFSWeights]={"Python"->Null,"Session"->Null,"Dir"->FileNameJoin[{$WORKDIR,"test"}]};
GetFSWeights[dataset_:"all",OptionsPattern[]]:=
Module[{python,session,outDir,res,pts,args},(
python=OptionValue["Python"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];
If[ListQ[dataset],
If[Length[dataset]==0,Return[{{},session}]];
If[Length[Dimensions[dataset]]==1,pts={Join[Re[dataset],Im[dataset[[;;]]]]},pts=Table[Join[Re[dataset[[i]]],Im[dataset[[i]]]],{i,Length[dataset]}]];
args="{
        'outdir':      \""<>ToPython[outDir]<>"\",
        'points':        "<>ToPython[pts]<>"
       }";
res=ExternalEvaluate[session,"mcy.get_weights"->args];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];
,
If[dataset=="all",
res=Join[
ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['y_train']"],ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['y_val']"]
];,
res=ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['y_"<>dataset<>"']"]
];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];
If[Length[res]>0,
res=res[[;;,-2]];
,
res={};
];
];
Return[{Normal[res],session}];
)];

Options[GetOmegaSquared]={"Python"->Null,"Session"->Null,"Dir"->FileNameJoin[{$WORKDIR,"test"}]};
GetOmegaSquared[dataset_:"all",OptionsPattern[]]:=
Module[{python,session,outDir,res,pts,args},(
python=OptionValue["Python"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];
If[ListQ[dataset],
If[Length[dataset]==0,Return[{{},session}]];
If[Length[Dimensions[dataset]]==1,pts={Join[Re[dataset],Im[dataset[[;;]]]]},pts=Table[Join[Re[dataset[[i]]],Im[dataset[[i]]]],{i,Length[dataset]}]];
args="{
        'outdir':      \""<>ToPython[outDir]<>"\",
        'points':        "<>ToPython[pts]<>"
       }";
res=ExternalEvaluate[session,"mcy.get_omegas"->args];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];
,
If[dataset=="all",
res=Join[
ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['y_train']"],ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['y_val']"]
];,
res=ExternalEvaluate[session,"import numpy as np;import os;data=np.load(os.path.join('"<>ToPython[outDir]<>"', 'dataset.npz'));data['y_"<>dataset<>"']"]
];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];
If[Length[res]>0,
res=res[[;;,-1]];
,
res={};
];
];
Return[{Re[Normal[res]],session}];
)];

Options[GetCYWeights]={"Python"->Null,"Session"->Null,"Model"->"MultFS","Kappa"->0.,"VolJNorm"->1.,"Dir"->FileNameJoin[{$WORKDIR,"test"}],"DimX"->3};
GetCYWeights[dataset_:"all",OptionsPattern[]]:=
Module[{python,outDir,session,res,pts,omegaSquared,gs,dets,dimX,tmp,i,model,kappa,volJNorm},(
python=OptionValue["Python"];
session=OptionValue["Session"];
kappa=OptionValue["Kappa"];
model=OptionValue["Model"];
outDir=OptionValue["Dir"];
dimX=OptionValue["DimX"];
volJNorm=OptionValue["VolJNorm"];
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];
If [ListQ[dataset],
If[Length[Dimensions[dataset]]==1,pts={dataset},pts=dataset];
,
{pts,tmp}=GetPoints[dataset,"Python"->python,"Session"->session,"Dir"->outDir];
];
{omegaSquared,tmp}=GetOmegaSquared[dataset,"Python"->python,"Session"->session,"Dir"->outDir];
{gs,tmp}=CYMetric[pts,"Python"->python,"Session"->session,"Model"->model,"Kappa"->kappa,"Dir"->outDir];
dets=Table[Det[gs[[i,2]]],{i,Length[gs]}];
dets=dets /volJNorm;
res=omegaSquared/dets;
Return[{Re[Normal[res]],session}];
)];

Options[GetPullbacks]={"Python"->Null,"Session"->Null,"Dir"->FileNameJoin[{$WORKDIR,"test"}]};
GetPullbacks[points_,OptionsPattern[]]:=
Module[{python,outDir,session,res,args,pts,kappa,model},(
python=OptionValue["Python"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];

If[Length[points]==0,Return[{{},session}]];
If[Length[Dimensions[points]]==1,pts={Join[Re[points],Im[points[[;;]]]]},pts=Table[Join[Re[points[[i]]],Im[points[[i]]]],{i,Length[points]}]];
args="{
        'outdir':      \""<>ToPython[outDir]<>"\",
        'points':        "<>ToPython[pts]<>"
       }";
res=ExternalEvaluate[session,"mcy.get_pullbacks"->args];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];
Return[{Normal[res],session}];
)];

Options[CYMetric]={"Python"->Null,"Session"->Null,"Model"->"MultFS","Kappa"->0.,"Dir"->FileNameJoin[{$WORKDIR,"test"}]};
CYMetric[points_,OptionsPattern[]]:=
Module[{python,outDir,session,res,args,pts,kappa,model},(
python=OptionValue["Python"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
kappa=OptionValue["Kappa"];
model=OptionValue["Model"];
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];

If[Length[points]==0,Return[{{},session}]];
If[Length[Dimensions[points]]==1,pts={Join[Re[points],Im[points[[;;]]]]},pts=Table[Join[Re[points[[i]]],Im[points[[i]]]],{i,Length[points]}]];
args="{
        'outdir':      \""<>ToPython[outDir]<>"\",
        'points':        "<>ToPython[pts]<>",
        'kappa':        "<>ToPython[kappa]<>",
        'model':        \""<>ToPython[model]<>"\"
       }";
res=ExternalEvaluate[session,"mcy.get_g"->args];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];
Return[{Normal[res],session}];
)];

Options[FSMetric]={"Python"->Null,"Session"->Null,"KahlerModuli"->{},"Dir"->FileNameJoin[{$WORKDIR,"test"}]};
FSMetric[points_,OptionsPattern[]]:=
Module[{python,outDir,session,res,args,pts,ts},(
python=OptionValue["Python"];
session=OptionValue["Session"];
outDir=OptionValue["Dir"];
ts=OptionValue["KahlerModuli"];
session=GetSession[python,session];
If[session===Null,Return[{"Could not start a Python Kernel with all dependencies installed.",session}]];

If[Length[points]==0,Return[{{},session}]];
If[Length[Dimensions[points]]==1,pts={Join[Re[points],Im[points[[;;]]]]},pts=Table[Join[Re[points[[i]]],Im[points[[i]]]],{i,Length[points]}]];
args="{
        'outdir':      \""<>ToPython[outDir]<>"\",
        'points':        "<>ToPython[pts]<>",
        'ts':            "<>ToPython[ts]<>"
       }";
res=ExternalEvaluate[session,"mcy.get_g_fs"->args];
If[FailureQ[res],Print["An error occurred."];Print[res];Return[{res,session}]];
Return[{Normal[res],session}];
)];
End[];


(* ::Input::Initialization:: *)
EndPackage[];
