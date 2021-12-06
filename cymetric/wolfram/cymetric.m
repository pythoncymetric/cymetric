(* Content-type: application/vnd.wolfram.mathematica *)

(*** Wolfram Notebook File ***)
(* http://www.wolfram.com/nb *)

(* CreatedBy='Mathematica 12.0' *)

(*CacheID: 234*)
(* Internal cache information:
NotebookFileLineBreakTest
NotebookFileLineBreakTest
NotebookDataPosition[       158,          7]
NotebookDataLength[    156752,       3295]
NotebookOptionsPosition[    155958,       3273]
NotebookOutlinePosition[    156405,       3290]
CellTagsIndexPosition[    156362,       3287]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{
Cell[BoxData[
 RowBox[{
  RowBox[{"BeginPackage", "[", 
   RowBox[{"\"\<cymetric\>\"", "`"}], "]"}], ";"}]], "Input",
 InitializationCell->True,
 CellChangeTimes->{{3.84781289284783*^9, 3.847812899615992*^9}, 
   3.8478129467267*^9},ExpressionUUID->"b7c0a472-a1ce-4fc3-84c0-7ff1a299e1b7"],

Cell[BoxData[
 RowBox[{"(*", "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
   "This", " ", "package", " ", "provides", " ", "a", " ", "Mathematica", " ",
     "interface", " ", "for", " ", "the", " ", "Python", " ", "cymetric", " ", 
    RowBox[{"paper", ".", " ", "Authors"}], " ", "Fabian", " ", "Ruehle", " ",
     "and", " ", "Robin", " ", 
    RowBox[{"Schneider", ".", " ", "For"}], " ", "more", " ", "information", 
    " ", "visit", " ", 
    RowBox[{"https", ":"}]}], "//", 
   RowBox[{
    RowBox[{
     RowBox[{"github", ".", "com"}], "/", "pythoncymetric"}], "/", 
    "cymetric"}]}], " ", "\[IndentingNewLine]", "*)"}]], "Input",
 InitializationCell->True,
 CellChangeTimes->{{3.847809425591614*^9, 
  3.847809511364319*^9}},ExpressionUUID->"0d1c205d-9948-4aa4-a8e4-\
3bd5ead1ac02"],

Cell[BoxData[{
 RowBox[{
  RowBox[{"$SETTINGSFILE", "=", 
   RowBox[{"FileNameJoin", "[", 
    RowBox[{"{", 
     RowBox[{
      RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<settings.txt\>\""}], 
     "}"}], "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetSetting", "::", "usage"}], "=", 
   "\"\<GetSetting[k] retrieves the value of k from $SETTINGSFILE.\\n* Input:\
\\n  - k (string): The entry whose value shall be retrieved\\n* Return:\\n  - \
The value v (string) of entry k, or Null if entry k not found\\n* Example:\\n \
 - GetSetting[\\\"Python\\\"]\>\""}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetSetting", "::", "fileNotFound"}], "=", 
   "\"\<Could not retrieve `1`. No settings file found.\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetSetting", "[", "k_", "]"}], ":=", 
   RowBox[{"Module", "[", 
    RowBox[{
     RowBox[{"{", "settings", "}"}], ",", 
     RowBox[{"(", "\[IndentingNewLine]", 
      RowBox[{
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"!", 
          RowBox[{"FileExistsQ", "[", "$SETTINGSFILE", "]"}]}], ",", 
         RowBox[{
          RowBox[{"Message", "[", 
           RowBox[{
            RowBox[{"GetSetting", "::", "fileNotFound"}], ",", "k"}], "]"}], 
          ";", 
          RowBox[{"Return", "[", "]"}], ";"}]}], "]"}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"settings", "=", 
        RowBox[{"ToExpression", "[", 
         RowBox[{"Import", "[", "$SETTINGSFILE", "]"}], "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"KeyExistsQ", "[", 
          RowBox[{"settings", ",", "k"}], "]"}], ",", 
         RowBox[{
          RowBox[{"Return", "[", 
           RowBox[{"settings", "[", "k", "]"}], "]"}], ";"}]}], "]"}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"Return", "[", "Null", "]"}], ";"}], "\[IndentingNewLine]", 
      ")"}]}], "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"ChangeSetting", "::", "usage"}], "=", 
   "\"\<ChangeSetting[k, v] sets the value of entry k to v in \
$SETTINGSFILE.\\n* Input:\\n  - k (string): The entry whose value shall be \
set\\n  - v (string): The value for entry k\\n* Return:\\n  - Null\\n* \
Example:\\n  - ChangeSetting[\\\"Python\\\",\\\"/usr/bin/python3\\\"]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"ChangeSetting", "[", 
    RowBox[{"k_", ",", "v_"}], "]"}], ":=", 
   RowBox[{"Module", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"settings", ",", "hnd"}], "}"}], ",", 
     RowBox[{"(", "\[IndentingNewLine]", 
      RowBox[{
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"!", 
          RowBox[{"FileExistsQ", "[", "$SETTINGSFILE", "]"}]}], ",", 
         "\[IndentingNewLine]", 
         RowBox[{
          RowBox[{"Export", "[", 
           RowBox[{"$SETTINGSFILE", ",", 
            RowBox[{"Association", "[", "]"}]}], "]"}], ";"}]}], 
        "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
       RowBox[{"settings", "=", 
        RowBox[{"ToExpression", "[", 
         RowBox[{"Import", "[", "$SETTINGSFILE", "]"}], "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"settings", "[", "k", "]"}], "=", "v"}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"hnd", "=", 
        RowBox[{"OpenWrite", "[", "$SETTINGSFILE", "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"Write", "[", 
        RowBox[{"hnd", ",", "settings"}], "]"}], ";", "\[IndentingNewLine]", 
       RowBox[{"Close", "[", "hnd", "]"}], ";", "\[IndentingNewLine]", 
       RowBox[{"Return", "[", "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], 
    "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"DeleteSetting", "::", "usage"}], "=", 
   "\"\<DeleteSetting[k] deletes the entry k in $SETTINGSFILE.\\n* Input:\\n  \
- k (string): The entry that shall be deleted\\n* Return:\\n  - Null\\n* \
Example:\\n  - DeleteSetting[\\\"Python\\\"]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"DeleteSetting", "[", "k_", "]"}], ":=", 
   RowBox[{"Module", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"settings", ",", "hnd"}], "}"}], ",", 
     RowBox[{"(", "\[IndentingNewLine]", 
      RowBox[{
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"!", 
          RowBox[{"FileExistsQ", "[", "$SETTINGSFILE", "]"}]}], ",", 
         "\[IndentingNewLine]", 
         RowBox[{
          RowBox[{"Export", "[", 
           RowBox[{"$SETTINGSFILE", ",", 
            RowBox[{"Association", "[", "]"}]}], "]"}], ";"}]}], 
        "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
       RowBox[{"settings", "=", 
        RowBox[{"ToExpression", "[", 
         RowBox[{"Import", "[", "$SETTINGSFILE", "]"}], "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"KeyExistsQ", "[", 
          RowBox[{"settings", ",", "k"}], "]"}], ",", 
         RowBox[{"KeyDropFrom", "[", 
          RowBox[{"settings", ",", "k"}], "]"}]}], "]"}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"hnd", "=", 
        RowBox[{"OpenWrite", "[", "$SETTINGSFILE", "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"Write", "[", 
        RowBox[{"hnd", ",", "settings"}], "]"}], ";", "\[IndentingNewLine]", 
       RowBox[{"Close", "[", "hnd", "]"}], ";", "\[IndentingNewLine]", 
       RowBox[{"Return", "[", "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], 
    "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"ToPython", "::", "usage"}], "=", 
   "\"\<ToPython[input] returns input in Python format.\\n* Input:\\n  - \
input (string or list): Mathematica expression\\n* Return:\\n  - input \
converted to Python syntax\\n* Example:\\n  - ToPython[{1.3 \[ImaginaryI]}]\>\
\""}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"ToPython", "[", "input_", "]"}], ":=", 
   RowBox[{"(", "\[IndentingNewLine]", 
    RowBox[{
     RowBox[{"SetOptions", "[", 
      RowBox[{"$Output", ",", 
       RowBox[{"PageWidth", "\[Rule]", "Infinity"}]}], "]"}], ";", 
     "\[IndentingNewLine]", 
     RowBox[{"If", "[", 
      RowBox[{
       RowBox[{"StringQ", "[", "input", "]"}], ",", 
       RowBox[{"Return", "[", "input", "]"}]}], "]"}], ";", 
     "\[IndentingNewLine]", 
     RowBox[{"Return", "[", 
      RowBox[{"StringReplace", "[", 
       RowBox[{
        RowBox[{"ToString", "[", 
         RowBox[{"InputForm", "[", 
          RowBox[{"input", ",", 
           RowBox[{"NumberMarks", "\[Rule]", "False"}]}], "]"}], "]"}], ",", 
        " ", 
        RowBox[{"{", 
         RowBox[{
          RowBox[{"\"\<{\>\"", "\[Rule]", "\"\<[\>\""}], ",", 
          RowBox[{"\"\<}\>\"", "\[Rule]", "\"\<]\>\""}], ",", " ", 
          RowBox[{"\"\<*^\>\"", "\[Rule]", "\"\<e\>\""}], ",", " ", 
          RowBox[{"\"\<^\>\"", "\[Rule]", "\"\<**\>\""}], ",", " ", 
          RowBox[{"\"\<I\>\"", "\[Rule]", "\"\<1.j\>\""}]}], "}"}]}], "]"}], 
      "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Setup", "::", "usage"}], "=", 
   "\"\<Setup[path, Options] finds a valid Python executable; if setup has \
been run before, it uses the path from global settings. Else it finds a good \
Python 3 interpreter and creates a virtual environment and patches \
mathematica to work with python >= 3.7.\\n* Input:\\n  - path (string): Sets \
up a python venv in path (defaults to ./venv).\\n* Options (run \
Options[Setup] to see default values):\\n  - ForceReinstall (bool): Whether \
the venv should be reinstalled if it already exists\\n* Return:\\n  - python \
(string): path to python venv executable\\n* Example:\\n  - Setup[]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "Setup", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{"\"\<ForceReinstall\>\"", "\[Rule]", "False"}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"Setup", "[", 
     RowBox[{
      RowBox[{"path_String", ":", 
       RowBox[{"FileNameJoin", "[", 
        RowBox[{"{", 
         RowBox[{
          RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<venv\>\""}], 
         "}"}], "]"}]}], ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "exec", ",", "res", ",", "settings", ",", "forceReinstall", ",", 
        "python", ",", " ", "packageDir"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"forceReinstall", "=", 
         RowBox[{"OptionValue", "[", "\"\<ForceReinstall\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{"Auto", "-", 
          RowBox[{
          "detect", " ", "whether", " ", "venv", " ", "is", " ", "already", 
           " ", "set", " ", "up", " ", "in", " ", "the", " ", "folder"}]}], 
         "*)"}], "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", "forceReinstall"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"python", "=", 
            RowBox[{"Quiet", "[", 
             RowBox[{"Check", "[", 
              RowBox[{
               RowBox[{"GetSetting", "[", "\"\<Python\>\"", "]"}], ",", 
               "Null"}], "]"}], "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"!", 
              RowBox[{"python", "===", "Null"}]}], ",", 
             RowBox[{
              RowBox[{"Return", "[", "python", "]"}], ";"}]}], "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FileExistsQ", "[", "$SETTINGSFILE", "]"}], ",", 
             RowBox[{
              RowBox[{
              "Print", "[", 
               "\"\<Settings file does not contain a path to a Python \
environment. Please set one with changeSettings[Python-><path/to/python>], or \
run again with option ForceReinstall->True\>\"", "]"}], ";"}]}], 
            "\[IndentingNewLine]", "]"}], ";"}]}], "\[IndentingNewLine]", 
         "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"exec", "=", 
         RowBox[{"DiscoverPython", "[", "True", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"SetupPythonVENV", "[", 
          RowBox[{"exec", ",", 
           RowBox[{"\"\<Patch\>\"", "\[Rule]", "True"}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"ChangeSetting", "[", 
         RowBox[{"\"\<Python\>\"", ",", "res"}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"packageDir", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{
          "\"\<Python\>\"", ",", 
           "\"\<import cymetric;os.path.dirname(cymetric.__file__)\>\""}], 
          "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
         "Import", " ", "the", " ", "mathematica", " ", "point", " ", 
          "generation", " ", "functions", " ", "into", " ", "the", " ", 
          "current", " ", "session"}], "*)"}], "\[IndentingNewLine]", 
        RowBox[{"Print", "[", 
         RowBox[{"FileNameJoin", "[", 
          RowBox[{"{", 
           RowBox[{
           "packageDir", ",", "\"\<wolfram/PointGeneratorMathematica.m\>\""}],
            "}"}], "]"}], "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Import", "[", 
         RowBox[{"FileNameJoin", "[", 
          RowBox[{"{", 
           RowBox[{
           "packageDir", ",", "\"\<wolfram/PointGeneratorMathematica.m\>\""}],
            "}"}], "]"}], "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Return", "[", "res", "]"}], ";"}], "\[IndentingNewLine]", 
       ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"DiscoverPython", "::", "usage"}], "=", 
   "\"\<DiscoverPython[useForVENV,Options] finds a Python executable for \
python3 with version <3.9.\\n* Input:\\n  - useForVENV (string): If False, \
keeps looking until it finds an environment with all necessary packages \
installed. Otherwise just uses the latest Python 3 version <3.9\\n* Options \
(run Options[DiscoverPython] to see default values):\\n  - Version (string): \
Specify a Python version to use \\n* Return:\\n  - python (string): path to \
python venv executable\\n* Example:\\n  - DiscoverPython[]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "DiscoverPython", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{"\"\<Version\>\"", "\[Rule]", "Null"}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"DiscoverPython", "[", 
     RowBox[{
      RowBox[{"useForVENV_", ":", "False"}], ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "pythonEnvs", ",", "exec", ",", "session", ",", "res", ",", "version", 
        ",", "i"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"version", "=", 
         RowBox[{"OptionValue", "[", "\"\<Version\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"pythonEnvs", "=", 
         RowBox[{"FindExternalEvaluators", "[", 
          RowBox[{"\"\<Python\>\"", ",", 
           RowBox[{"\"\<ResetCache\>\"", "\[Rule]", "True"}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"pythonEnvs", "=", 
         RowBox[{"Reverse", "[", 
          RowBox[{"SortBy", "[", 
           RowBox[{"pythonEnvs", ",", "\"\<Version\>\""}], "]"}], "]"}]}], 
        ";", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{
          "Start", " ", "with", " ", "latest", " ", "Python", " ", "on", " ", 
           "tyhe", " ", "system", " ", "and", " ", "work", " ", "your", " ", 
           "way", " ", 
           RowBox[{"down", ".", " ", "ATM"}]}], ",", " ", 
          RowBox[{
           RowBox[{"there", "'"}], "s", " ", "no", " ", "TF", " ", "for", " ",
            "python", " ", "3.9"}], ",", " ", 
          RowBox[{"so", " ", "we", " ", "skip", " ", "that"}]}], "*)"}], 
        "\[IndentingNewLine]", 
        RowBox[{
        "Print", "[", 
         "\"\<Mathematica discovered the following Python environments on \
your system:\>\"", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Print", "[", "pythonEnvs", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Print", "[", "\"\<Looking for Python 3\>\"", "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"For", " ", "[", 
         RowBox[{
          RowBox[{"i", "=", "1"}], ",", 
          RowBox[{"i", "\[LessEqual]", 
           RowBox[{"Length", "[", "pythonEnvs", "]"}]}], ",", 
          RowBox[{"i", "++"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{
               RowBox[{"StringTake", "[", 
                RowBox[{
                 RowBox[{
                  RowBox[{"pythonEnvs", "[", "i", "]"}], "[", 
                  "\"\<Version\>\"", "]"}], ",", "1"}], "]"}], "\[NotEqual]", 
               "\"\<3\>\""}], "||", 
              RowBox[{
               RowBox[{"StringTake", "[", 
                RowBox[{
                 RowBox[{
                  RowBox[{"pythonEnvs", "[", "i", "]"}], "[", 
                  "\"\<Version\>\"", "]"}], ",", "3"}], "]"}], "\[Equal]", 
               "\"\<3.9\>\""}]}], ",", " ", 
             RowBox[{"Continue", "[", "]"}]}], "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"!", 
              RowBox[{"version", "===", "Null"}]}], ",", 
             RowBox[{"If", "[", 
              RowBox[{
               RowBox[{"version", "\[NotEqual]", 
                RowBox[{
                 RowBox[{"pythonEnvs", "[", "i", "]"}], "[", 
                 "\"\<Version\>\"", "]"}]}], ",", 
               RowBox[{
                RowBox[{"Continue", "[", "]"}], ";"}]}], "]"}]}], "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"exec", "=", 
            RowBox[{
             RowBox[{"pythonEnvs", "[", "i", "]"}], "[", "\"\<Executable\>\"",
              "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"Print", "[", 
            RowBox[{"\"\<Found Python version \>\"", ",", 
             RowBox[{
              RowBox[{"pythonEnvs", "[", "i", "]"}], "[", "\"\<Version\>\"", 
              "]"}], ",", " ", "\"\< at \>\"", ",", "exec", ",", 
             "\"\<.\>\""}], "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{"useForVENV", ",", 
             RowBox[{"Return", "[", "exec", "]"}]}], "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{
           "Print", "[", 
            "\"\<Looking for pyzmq for Mathematica interop and \
tensorflow...\>\"", "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"session", "=", 
            RowBox[{"Quiet", "[", 
             RowBox[{"Check", "[", 
              RowBox[{
               RowBox[{"StartExternalSession", "[", 
                RowBox[{"<|", 
                 RowBox[{
                  RowBox[{"\"\<System\>\"", "\[Rule]", "\"\<Python\>\""}], 
                  ",", 
                  RowBox[{"\"\<Executable\>\"", "\[Rule]", "exec"}]}], "|>"}],
                 "]"}], ",", 
               RowBox[{
                RowBox[{
                "Print", "[", 
                 "\"\<Mathematica couldn't start this Python session. Trying \
next Python environment...\>\"", "]"}], ";", 
                RowBox[{"Continue", "[", "]"}]}]}], "]"}], "]"}]}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"DeleteObject", "[", "session", "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"(*", 
            RowBox[{"Attempt", " ", "to", " ", "load"}], "*)"}], 
           "\[IndentingNewLine]", 
           RowBox[{"session", "=", 
            RowBox[{"GetSession", "[", "exec", "]"}]}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"session", "===", "Null"}], ",", 
             RowBox[{
              RowBox[{"Continue", "[", "]"}], ";"}], ",", 
             RowBox[{"Print", "[", 
              RowBox[{
              "\"\<Found good Python environment: \>\"", ",", "exec"}], 
              "]"}]}], "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"Quiet", "[", 
            RowBox[{"Check", "[", 
             RowBox[{
              RowBox[{"DeleteObject", "[", "session", "]"}], ",", 
              RowBox[{"Continue", "[", "]"}]}], "]"}], "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"Return", "[", "exec", "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], 
     "]"}]}], ";"}], "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"SetupPythonVENV", "::", "usage"}], "=", 
   "\"\<SetupPythonVENV[exec,Options] sets up a Python virtual \
environment.\\n* Input:\\n  - exec (string): Path to Python that will be used \
to create the venv.\\n* Options (run Options[SetupPythonVENV] to see default \
values):\\n  - VENVPath (string): Path where venv will be created\\n  - Patch \
(bool): Whether Mathematica should be patched to fix a bug for Python >3.5 \
(this only changes a text file used for Python interop and does not interfere \
with the Mathematica installation) \\n* Return:\\n  - python (string): path \
to python venv executable\\n* Example:\\n  - \
SetupPythonVENV[\\\"/usr/bin/python3\\\"]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "SetupPythonVENV", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<VENVPath\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<venv\>\""}], 
        "}"}], "]"}]}], ",", 
     RowBox[{"\"\<Patch\>\"", "\[Rule]", "False"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"SetupPythonVENV", "[", 
     RowBox[{"exec_String", ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "path", ",", "patchEE", ",", "setupVENV", ",", "python", ",", "pip", 
        ",", "installPackages", ",", "packages", ",", "session", ",", "res", 
        ",", "hnd", ",", "content", ",", "i"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"path", "=", 
         RowBox[{"OptionValue", "[", "\"\<VENVPath\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"patchEE", "=", 
         RowBox[{"OptionValue", "[", "\"\<Patch\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Print", "[", 
         RowBox[{"\"\<Creating virtual environment at \>\"", ",", "path"}], 
         "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"setupVENV", "=", 
         RowBox[{"RunProcess", "[", 
          RowBox[{"{", 
           RowBox[{
           "exec", ",", "\"\<-m\>\"", ",", "\"\<venv\>\"", ",", "path"}], 
           "}"}], "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"setupVENV", "[", "\"\<ExitCode\>\"", "]"}], "\[NotEqual]",
            "0"}], ",", 
          RowBox[{
           RowBox[{
           "Print", "[", "\"\<An error occurred. Here's the output\>\"", 
            "]"}], ";", 
           RowBox[{"Print", "[", 
            RowBox[{"setupVENV", "[", "\"\<StandardOutput\>\"", "]"}], "]"}], 
           ";", 
           RowBox[{"Print", "[", 
            RowBox[{"setupVENV", "[", "\"\<StandardError\>\"", "]"}], "]"}], 
           ";", 
           RowBox[{"Return", "[", "]"}], ";"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"python", "=", 
         RowBox[{"FileNameJoin", "[", 
          RowBox[{"{", 
           RowBox[{"path", ",", "\"\<bin\>\"", ",", "\"\<python\>\""}], "}"}],
           "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"pip", "=", 
         RowBox[{"FileNameJoin", "[", 
          RowBox[{"{", 
           RowBox[{"path", ",", "\"\<bin\>\"", ",", "\"\<pip\>\""}], "}"}], 
          "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", 
           RowBox[{"FileExistsQ", "[", "pip", "]"}]}], ",", 
          RowBox[{
           RowBox[{"pip", "=", 
            RowBox[{"FileNameJoin", "[", 
             RowBox[{"{", 
              RowBox[{"path", ",", "\"\<bin\>\"", ",", "\"\<pip3\>\""}], 
              "}"}], "]"}]}], ";"}]}], "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", 
           RowBox[{"FileExistsQ", "[", "pip", "]"}]}], ",", 
          RowBox[{
           RowBox[{"Print", "[", 
            RowBox[{"\"\<Error: Couldn't find pip at \>\"", ",", 
             RowBox[{"FileNameJoin", "[", 
              RowBox[{"{", 
               RowBox[{"path", ",", "\"\<bin\>\""}], "}"}], "]"}]}], "]"}], 
           ";", 
           RowBox[{"Return", "[", "python", "]"}], ";"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{"Install", " ", "packages"}], "*)"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"packages", "=", 
         RowBox[{"{", " ", 
          RowBox[{
          "\"\<h5py\>\"", ",", "\"\<joblib\>\"", ",", "\"\<numpy\>\"", ",", 
           "\"\<pyyaml\>\"", ",", "\"\<pyzmq\>\"", ",", "\"\<scipy\>\"", ",", 
           "\"\<sympy\>\"", ",", "\"\<tensorflow==2.4.1\>\"", ",", 
           "\"\<wolframclient\>\""}], "}"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"Print", "[", "\"\<Upgrading pip...\>\"", "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"installPackages", "=", 
         RowBox[{"RunProcess", "[", 
          RowBox[{"{", 
           RowBox[{
           "pip", ",", "\"\<install\>\"", ",", " ", "\"\<--upgrade\>\"", ",", 
            " ", "\"\<pip\>\""}], "}"}], "]"}]}], ";", "\[IndentingNewLine]", 
        
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"installPackages", "[", "\"\<ExitCode\>\"", "]"}], 
           "\[NotEqual]", "0"}], ",", 
          RowBox[{
           RowBox[{
           "Print", "[", "\"\<An error occurred. Here's the output\>\"", 
            "]"}], ";", 
           RowBox[{"Print", "[", 
            RowBox[{"installPackages", "[", "\"\<StandardOutput\>\"", "]"}], 
            "]"}], ";", 
           RowBox[{"Print", "[", 
            RowBox[{"installPackages", "[", "\"\<StandardError\>\"", "]"}], 
            "]"}], ";", 
           RowBox[{"Return", "[", "python", "]"}], ";"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"For", "[", 
         RowBox[{
          RowBox[{"i", "=", "1"}], ",", 
          RowBox[{"i", "\[LessEqual]", 
           RowBox[{"Length", "[", "packages", "]"}]}], ",", 
          RowBox[{"i", "++"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"Print", "[", 
            RowBox[{"\"\<Installing \>\"", ",", 
             RowBox[{"packages", "[", 
              RowBox[{"[", "i", "]"}], "]"}], ",", "\"\<...\>\""}], "]"}], 
           ";", "\[IndentingNewLine]", 
           RowBox[{"installPackages", "=", 
            RowBox[{"RunProcess", "[", 
             RowBox[{"{", 
              RowBox[{"pip", ",", "\"\<install\>\"", ",", 
               RowBox[{"packages", "[", 
                RowBox[{"[", "i", "]"}], "]"}]}], "}"}], "]"}]}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"installPackages", "[", "\"\<ExitCode\>\"", "]"}], 
              "\[NotEqual]", "0"}], ",", 
             RowBox[{
              RowBox[{
              "Print", "[", "\"\<An error occurred. Here's the output\>\"", 
               "]"}], ";", 
              RowBox[{"Print", "[", 
               RowBox[{
               "installPackages", "[", "\"\<StandardOutput\>\"", "]"}], "]"}],
               ";", 
              RowBox[{"Print", "[", 
               RowBox[{"installPackages", "[", "\"\<StandardError\>\"", "]"}],
                "]"}], ";", 
              RowBox[{"Return", "[", "python", "]"}], ";"}]}], 
            "\[IndentingNewLine]", "]"}], ";"}]}], "\[IndentingNewLine]", 
         "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{"Install", " ", "CY", " ", "package"}], "*)"}], 
        "\[IndentingNewLine]", 
        RowBox[{"Print", "[", "\"\<Installing cymetric...\>\"", "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"installPackages", "=", 
         RowBox[{
         "RunProcess", "[", 
          "\"\<pip install git+https://github.com/pythoncymetric/cymetric.git\
\>\"", "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"installPackages", "[", "\"\<ExitCode\>\"", "]"}], 
           "\[NotEqual]", "0"}], ",", 
          RowBox[{
           RowBox[{
           "Print", "[", "\"\<An error occurred. Here's the output\>\"", 
            "]"}], ";", 
           RowBox[{"Print", "[", 
            RowBox[{"installPackages", "[", "\"\<StandardOutput\>\"", "]"}], 
            "]"}], ";", 
           RowBox[{"Print", "[", 
            RowBox[{"installPackages", "[", "\"\<StandardError\>\"", "]"}], 
            "]"}], ";", 
           RowBox[{"Return", "[", "python", "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
         "Register", " ", "for", " ", "use", " ", "with", " ", 
          "Mathematica"}], "*)"}], "\[IndentingNewLine]", 
        RowBox[{
        "Print", "[", "\"\<Registering venv with mathematica...\>\"", "]"}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"RegisterExternalEvaluator", "[", 
         RowBox[{"\"\<Python\>\"", ",", "python"}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{
        "Print", "[", 
         "\"\<Checking whether 'externalevaluate.py' needs to be patched for \
this version...\>\"", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"StartExternalSession", "[", 
          RowBox[{"<|", 
           RowBox[{
            RowBox[{"\"\<System\>\"", "\[Rule]", "\"\<Python\>\""}], ",", 
            RowBox[{"\"\<Executable\>\"", "\[Rule]", "python"}]}], "|>"}], 
          "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"Quiet", "[", 
          RowBox[{"ExternalEvaluate", "[", 
           RowBox[{"session", ",", 
            RowBox[{"{", "\"\<import os;x=0;x\>\"", "}"}]}], "]"}], "]"}]}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"If", " ", "[", 
         RowBox[{
          RowBox[{"ListQ", "[", "res", "]"}], ",", 
          RowBox[{"res", "=", " ", 
           RowBox[{"res", "[", 
            RowBox[{"[", "1", "]"}], "]"}]}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"res", "[", "\"\<Message\>\"", "]"}], "\[Equal]", 
           "\"\<required field \\\"type_ignores\\\" missing from \
Module\>\""}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<Patch needs to be applied\>\"", "]"}], 
           ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"!", "patchEE"}], ",", "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{
              "Print", "[", 
               "\"\<Option for automatically applying patch not set. Please \
apply manually and try again.\>\"", "]"}], ";", "\[IndentingNewLine]", 
              RowBox[{"DeleteObject", "[", "session", "]"}], ";", 
              "\[IndentingNewLine]", 
              RowBox[{"Return", "[", "python", "]"}], ";"}], 
             "\[IndentingNewLine]", ",", "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"Print", "[", "\"\<Applying patch...\>\"", "]"}], ";", 
              "\[IndentingNewLine]", 
              RowBox[{"hnd", "=", 
               RowBox[{"OpenRead", "[", 
                RowBox[{"FileNameJoin", "[", 
                 RowBox[{"{", " ", 
                  RowBox[{
                  "$InstallationDirectory", ",", 
                   "\"\</SystemFiles/Links/WolframClientForPython/\
wolframclient/utils/externalevaluate.py\>\""}], "}"}], "]"}], "]"}]}], ";", 
              "\[IndentingNewLine]", 
              RowBox[{"content", "=", 
               RowBox[{"ReadString", "[", "hnd", "]"}]}], ";", 
              "\[IndentingNewLine]", 
              RowBox[{"Close", "[", "hnd", "]"}], ";", "\[IndentingNewLine]", 
              
              RowBox[{"content", "=", 
               RowBox[{"StringReplace", "[", 
                RowBox[{"content", ",", 
                 RowBox[{"{", 
                  RowBox[{
                  "\"\<exec(compile(ast.Module(expressions), '', 'exec'), \
current)\>\"", "\[Rule]", 
                   "\"\<exec(compile(ast.Module(expressions, []), '', \
'exec'), current)  # exec(compile(ast.Module(expressions), '', 'exec'), \
current) # changed by CYMetrics\>\""}], "}"}]}], "]"}]}], ";", 
              "\[IndentingNewLine]", 
              RowBox[{"hnd", "=", 
               RowBox[{"OpenWrite", "[", 
                RowBox[{"FileNameJoin", "[", 
                 RowBox[{"{", " ", 
                  RowBox[{
                  "$InstallationDirectory", ",", 
                   "\"\</SystemFiles/Links/WolframClientForPython/\
wolframclient/utils/externalevaluate.py\>\""}], "}"}], "]"}], "]"}]}], ";", 
              "\[IndentingNewLine]", 
              RowBox[{"WriteString", "[", 
               RowBox[{"hnd", ",", "content"}], "]"}], ";", 
              "\[IndentingNewLine]", 
              RowBox[{"Close", "[", "hnd", "]"}], ";"}]}], 
            "\[IndentingNewLine]", "]"}], ";"}]}], "\[IndentingNewLine]", 
         "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Print", "[", "\"\<Testing new environment...\>\"", "]"}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", "python", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", 
           RowBox[{"session", "===", "Null"}]}], ",", 
          RowBox[{"DeleteObject", "[", "session", "]"}], ",", 
          RowBox[{"Return", "[", "python", "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Print", "[", "\"\<Everything is working!\>\"", "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Return", "[", "python", "]"}], ";"}], "\[IndentingNewLine]", 
       ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetSession", "::", "usage"}], "=", 
   "\"\<GetSession[exec,mysession] returns a Python session with the cymetric \
package loaded.\\n* Input:\\n  - exec (string): Path to Python that will be \
used to create the session. If not specified, the installation from \
$SETTINGSFILE is used\\n  - mysession (ExternalSessionObject): If mysession \
is a valid Python session, the cypackage and all dependencies are loaded \
intyo the session if necessary \\n* Return:\\n  - session \
(ExternalSessionObject): session with cymetric package and all dependencies \
loaded\\n* Example:\\n  - GetSession[]\>\""}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GetSession", "[", 
     RowBox[{
      RowBox[{"exec_", ":", "Null"}], ",", 
      RowBox[{"mysession_", ":", "Null"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{"validSession", ",", "session", ",", "python", ",", "res"}], 
       "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", "exec"}], ";", "\[IndentingNewLine]", 
        RowBox[{"session", "=", "mysession"}], ";", "\[IndentingNewLine]", 
        RowBox[{"validSession", "=", 
         RowBox[{"Quiet", "[", 
          RowBox[{"Check", "[", 
           RowBox[{
            RowBox[{"ExternalEvaluate", "[", 
             RowBox[{"session", ",", "\"\<2+2==4\>\""}], "]"}], ",", 
            "False"}], "]"}], "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"session", "===", "Null"}], "||", 
           RowBox[{"!", "validSession"}]}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"python", "===", "Null"}], ",", 
             RowBox[{"python", "=", 
              RowBox[{"Check", "[", 
               RowBox[{
                RowBox[{"GetSetting", "[", "\"\<Python\>\"", "]"}], ",", 
                "Null"}], "]"}]}]}], "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"python", "===", "Null"}], ",", 
             RowBox[{
              RowBox[{
              "Print", "[", 
               "\"\<No python interpreter configured. Please provide one with \
the option \\\"Python-><path/to/python>\\\"\>\"", "]"}], ";", 
              RowBox[{"Return", "[", "Null", "]"}], ";"}]}], "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"(*", 
            RowBox[{
             RowBox[{"Start", " ", "session"}], ",", " ", 
             RowBox[{
             "load", " ", "in", " ", "prepare", " ", "data", " ", "file"}]}], 
            "*)"}], "\[IndentingNewLine]", 
           RowBox[{"session", "=", 
            RowBox[{"StartExternalSession", "[", 
             RowBox[{"<|", 
              RowBox[{
               RowBox[{"\"\<System\>\"", "\[Rule]", "\"\<Python\>\""}], ",", 
               RowBox[{"\"\<Executable\>\"", "\[Rule]", "python"}]}], "|>"}], 
             "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"res", "=", 
            RowBox[{"ExternalEvaluate", "[", 
             RowBox[{
             "session", ",", 
              "\"\<import cymetric;import cymetric.wolfram.mathematicalib as \
mcy;\>\""}], "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FailureQ", "[", "res", "]"}], ",", " ", 
             RowBox[{
              RowBox[{"Print", "[", "res", "]"}], ";", 
              RowBox[{"Return", "[", "Null", "]"}], ";"}]}], "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{
           RowBox[{"Check", " ", "whether", " ", 
            RowBox[{"mathematica_lib", ".", "py"}], " ", "has", " ", "been", 
            " ", "loaded"}], ";", " ", 
           RowBox[{"if", " ", "not"}]}], ",", " ", 
          RowBox[{"load", " ", "it", " ", "into", " ", "session"}]}], "*)"}], 
        "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{
          "session", ",", "\"\<import sys;'cymetric' in sys.modules\>\""}], 
          "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", " ", 
          RowBox[{
           RowBox[{"Print", "[", "res", "]"}], ";", 
           RowBox[{"Return", "[", "Null", "]"}], ";"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", "res"}], ",", 
          RowBox[{"res", "=", 
           RowBox[{"ExternalEvaluate", "[", 
            RowBox[{
            "session", ",", 
             "\"\<import cymetric;import cymetric.wolfram.mathematicalib as \
mcy;\>\""}], "]"}]}]}], "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", " ", 
          RowBox[{
           RowBox[{"Print", "[", "res", "]"}], ";", 
           RowBox[{"Return", "[", "Null", "]"}], ";"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Return", "[", "session", "]"}], ";"}], "\[IndentingNewLine]",
        ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GeneratePoints", "::", "usage"}], "=", 
   "\"\<GeneratePoints[poly,dimPs,variables,Options] generates points on the \
CY specified by poly in an ambient space of dimPs.\\n* Input:\\n  - poly \
(list of polynomials): list of polynomials that define the CY.\\n  - dimPs \
(list of ints): list of dimensions of the product of projective ambient \
spaces.\\n  - variables [optional] (list of list of vars): list of list of \
variables (one for each projective ambient space). If not provided, \
alphabetical ordering is assumed.\\n* Options (run Options[GeneratePoints] to \
see default values):\\n  - KahlerModuli (list of floats): Kahler moduli \
\!\(\*SubscriptBox[\(t\), \(i\)]\) of the i'th ambient space factor (defaults \
to all 1)\\n  - Points (int): Number of points to generate (defaults to \
200,000)\\n  - Precision (int): WorkingPrecision to use when generating the \
points (defaults to 20)\\n  - VolJNorm (float): Normalization for the volume, \
input int_X J^n at t1=t2=...=1 (defaults to 1)\\n  - Python (string): python \
executable to use (defaults to the one of $SETTINGSFILE)\\n  - Session \
(ExternalSessionObject): Python session with all dependencies loaded (if not \
provided, a new one will be generated)\\n  - Dir (string): Directory where \
points will be saved\\n  - Verbose (int): Verbose level (the higher, the more \
info is printed)\\n* Return:\\n  - res (object): Null if no error occured, \
otherwise a string with the error\\n  - session (ExternalSessionObject): The \
session object used in the coomputation (for potential faster future reuse) \
\\n* Example:\\n  - GeneratePoints[{\!\(\*SuperscriptBox[SubscriptBox[\(z\), \
\(0\)], \(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(1\)], \(5\)]\)+\!\
\(\*SuperscriptBox[SubscriptBox[\(z\), \(2\)], \
\(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(3\)], \
\(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(4\)], \
\(5\)]\)},{4},\\\"Points\\\"\[Rule]200000,\\\"KahlerModuli\\\"\[Rule]{1},\\\"\
Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "GeneratePoints", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<KahlerModuli\>\"", "\[Rule]", 
      RowBox[{"{", "}"}]}], ",", 
     RowBox[{"\"\<Points\>\"", "\[Rule]", "200000"}], ",", 
     RowBox[{"\"\<Precision\>\"", "\[Rule]", "20"}], ",", 
     RowBox[{"\"\<VolJNorm\>\"", "\[Rule]", "1"}], ",", 
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}], ",", 
     RowBox[{"\"\<Verbose\>\"", "\[Rule]", "3"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GeneratePoints", "[", 
     RowBox[{"poly_", ",", "dimPs_", ",", 
      RowBox[{"variables_List", ":", 
       RowBox[{"{", "}"}]}], ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "points", ",", "numPts", ",", "pointsFile", ",", 
        "verbose", ",", "session", ",", "outDir", ",", "res", ",", "startPos",
         ",", " ", "monomials", ",", " ", "coeffs", ",", "kahlerModuli", ",", 
        " ", "precision", ",", "loggerLevel", ",", "args", ",", "vars", ",", 
        "randomPoint", ",", "i", ",", "prev", ",", "curr", ",", "isSymmetric",
         ",", "pointsBatched", ",", "volJNorm", ",", "numParamsInPn"}], "}"}],
       ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"numPts", "=", 
         RowBox[{"OptionValue", "[", "\"\<Points\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"verbose", "=", 
         RowBox[{"OptionValue", "[", "\"\<Verbose\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"kahlerModuli", "=", 
         RowBox[{"OptionValue", "[", "\"\<KahlerModuli\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"precision", "=", 
         RowBox[{"OptionValue", "[", "\"\<Precision\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"volJNorm", "=", 
         RowBox[{"OptionValue", "[", "\"\<VolJNorm\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{"Start", " ", "session"}], ",", " ", 
          RowBox[{
          "load", " ", "in", " ", "prepare", " ", "data", " ", "file"}]}], 
         "*)"}], "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", 
          RowBox[{"python", ",", "session"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"loggerLevel", "=", "\"\<ERROR\>\""}], ";", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{"Taken", " ", "for", " ", "verbose"}], " ", "\[LessEqual]", 
          " ", "0"}], "*)"}], "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[Equal]", "1"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<WARNING\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[Equal]", "2"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<INFO\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[GreaterEqual]", "3"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<DEBUG\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
         "Find", " ", "coefficients", " ", "and", " ", "exponents", " ", "of",
           " ", "the", " ", "polynomials"}], "*)"}], "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", 
           RowBox[{"ListQ", "[", "poly", "]"}]}], ",", 
          RowBox[{"poly", "=", 
           RowBox[{"{", "poly", "}"}]}]}], "]"}], ";", "\[IndentingNewLine]", 
        
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"variables", "\[Equal]", 
           RowBox[{"{", "}"}]}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"verbose", " ", ">", "0"}], ",", 
             RowBox[{
             "Print", "[", 
              "\"\<Warning: No variables specified, assuming alphabetical \
monomial ordering.\>\"", "]"}]}], "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"vars", "=", 
            RowBox[{"Sort", "[", 
             RowBox[{"Variables", "[", 
              RowBox[{"Flatten", "[", "poly", "]"}], "]"}], "]"}]}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"verbose", " ", ">", "1"}], ",", "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{
              "Print", "[", 
               "\"\<Variables have been assigned to the ambient space factors \
as follows:\>\"", "]"}], ";", "\[IndentingNewLine]", 
              RowBox[{"startPos", "=", "1"}], ";", "\[IndentingNewLine]", 
              RowBox[{"For", "[", 
               RowBox[{
                RowBox[{"i", "=", "1"}], ",", 
                RowBox[{"i", "\[LessEqual]", 
                 RowBox[{"Length", "[", "dimPs", "]"}]}], ",", 
                RowBox[{"i", "++"}], ",", "\[IndentingNewLine]", 
                RowBox[{
                 RowBox[{"Print", "[", 
                  RowBox[{"i", ",", "\"\<.) \>\"", ",", 
                   SuperscriptBox["P", 
                    RowBox[{"ToString", "[", 
                    RowBox[{"dimPs", "[", 
                    RowBox[{"[", "i", "]"}], "]"}], "]"}]], ",", "\"\<: \>\"",
                    ",", 
                   RowBox[{"vars", "[", 
                    RowBox[{"[", 
                    RowBox[{"startPos", ";;", 
                    RowBox[{"startPos", "+", 
                    RowBox[{"dimPs", "[", 
                    RowBox[{"[", "i", "]"}], "]"}]}]}], "]"}], "]"}]}], "]"}],
                  ";", "\[IndentingNewLine]", 
                 RowBox[{"startPos", "+=", 
                  RowBox[{
                   RowBox[{"dimPs", "[", 
                    RowBox[{"[", "i", "]"}], "]"}], "+", "1"}]}], ";"}]}], 
               "\[IndentingNewLine]", "]"}], ";"}], "\[IndentingNewLine]", 
             ",", "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"vars", "=", 
               RowBox[{"Flatten", "[", "vars", "]"}]}], ";"}]}], 
            "\[IndentingNewLine]", "]"}], ";"}], "\[IndentingNewLine]", ",", 
          "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"vars", "=", 
            RowBox[{"Flatten", "[", "variables", "]"}]}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"monomials", "=", 
         RowBox[{"{", "}"}]}], ";", 
        RowBox[{"coeffs", "=", 
         RowBox[{"{", "}"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"For", "[", 
         RowBox[{
          RowBox[{"i", "=", "1"}], ",", 
          RowBox[{"i", "\[LessEqual]", 
           RowBox[{"Length", "[", "poly", "]"}]}], ",", 
          RowBox[{"i", "++"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"res", "=", 
            RowBox[{"Association", "[", 
             RowBox[{"CoefficientRules", "[", 
              RowBox[{
               RowBox[{"poly", "[", 
                RowBox[{"[", "i", "]"}], "]"}], ",", "vars"}], "]"}], "]"}]}],
            ";", "\[IndentingNewLine]", 
           RowBox[{"AppendTo", "[", 
            RowBox[{"monomials", ",", 
             RowBox[{"Keys", "[", "res", "]"}]}], "]"}], ";", 
           "\[IndentingNewLine]", 
           RowBox[{"AppendTo", "[", 
            RowBox[{"coeffs", " ", ",", 
             RowBox[{"Values", "[", "res", "]"}]}], "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{"If", "[", 
           RowBox[{
            RowBox[{
             RowBox[{"Length", "[", "poly", "]"}], "\[Equal]", "1"}], ",", 
            RowBox[{
             RowBox[{"monomials", "=", 
              RowBox[{"monomials", "[", 
               RowBox[{"[", "1", "]"}], "]"}]}], ";", 
             RowBox[{"coeffs", "=", 
              RowBox[{"coeffs", "[", 
               RowBox[{"[", "1", "]"}], "]"}]}]}]}], "]"}], ";"}], "*)"}], 
        "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{"monomials", ",", "coeffs"}], "}"}], "]"}], ";"}], "*)"}],
         "\[IndentingNewLine]", 
        RowBox[{"kahlerModuli", " ", "=", " ", 
         RowBox[{"If", "[", 
          RowBox[{
           RowBox[{"kahlerModuli", "\[Equal]", 
            RowBox[{"{", "}"}]}], ",", " ", 
           RowBox[{"Table", "[", 
            RowBox[{"1", ",", 
             RowBox[{"{", 
              RowBox[{"i", ",", 
               RowBox[{"Length", "[", "dimPs", "]"}]}], "}"}]}], "]"}], ",", 
           "kahlerModuli"}], "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"pointsFile", "=", 
         RowBox[{"FileNameJoin", "[", 
          RowBox[{"{", 
           RowBox[{"outDir", ",", "\"\<points.pickle\>\""}], "}"}], "]"}]}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"Print", "[", 
         RowBox[{
         "\"\<Generating \>\"", ",", "numPts", ",", "\"\< points...\>\""}], 
         "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{
         RowBox[{"{", 
          RowBox[{"points", ",", "numParamsInPn"}], "}"}], "=", 
         RowBox[{"GeneratePointsM", "[", 
          RowBox[{
          "numPts", ",", "dimPs", ",", "coeffs", ",", "monomials", ",", 
           "precision", ",", "verbose", ",", "True"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Print", "[", 
         RowBox[{"\"\<Writing points to \>\"", ",", "pointsFile"}], "]"}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", 
           RowBox[{"DirectoryQ", "[", "outDir", "]"}]}], ",", 
          RowBox[{"CreateDirectory", "[", "outDir", "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"(*", " ", 
         RowBox[{
          RowBox[{"batch", " ", "the", " ", "conversion"}], ",", " ", 
          RowBox[{
          "otherwise", " ", "there", " ", "is", " ", "an", " ", "error", " ", 
           "for", " ", "long", " ", "points"}]}], "*)"}], 
        "\[IndentingNewLine]", 
        RowBox[{"pointsBatched", "=", 
         RowBox[{"Partition", "[", 
          RowBox[{"points", ",", 
           RowBox[{"UpTo", "[", "1000", "]"}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"ExternalEvaluate", "[", 
         RowBox[{"session", ",", "\"\<generated_pts=[];\>\""}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"For", "[", 
         RowBox[{
          RowBox[{"i", "=", "1"}], ",", 
          RowBox[{"i", "\[LessEqual]", 
           RowBox[{"Length", "[", "pointsBatched", "]"}]}], ",", 
          RowBox[{"i", "++"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"res", "=", 
            RowBox[{"ExternalEvaluate", "[", 
             RowBox[{"session", ",", 
              RowBox[{"\"\<generated_pts +=\>\"", "<>", 
               RowBox[{"ToPython", "[", 
                RowBox[{"pointsBatched", "[", 
                 RowBox[{"[", "i", "]"}], "]"}], "]"}], "<>", "\"\<;\>\""}]}],
              "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FailureQ", "[", "res", "]"}], ",", 
             RowBox[{
              RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
              
              RowBox[{"Print", "[", "res", "]"}], ";", 
              RowBox[{"Break", "[", "]"}], ";"}]}], "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{"session", ",", 
           RowBox[{
           "\"\<import pickle;import numpy as \
np;pickle.dump(np.array(generated_pts), open(\\\"\>\"", "<>", "pointsFile", 
            "<>", "\"\<\\\",'wb'));\>\""}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"ExternalEvaluate", "[", 
         RowBox[{"session", ",", "\"\<generated_pts=[];\>\""}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";"}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"args", "=", 
         RowBox[{"\"\<{\n        'outdir':        \\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
          "\"\<\\\",\n        'logger_level':  logging.\>\"", "<>", 
          "loggerLevel", "<>", "\"\<,\n        'num_pts':       \>\"", "<>", 
          RowBox[{"ToPython", "[", "numPts", "]"}], "<>", 
          "\"\<,\n        'monomials':     \>\"", "<>", 
          RowBox[{"ToPython", "[", "monomials", "]"}], "<>", 
          "\"\<,       \n        'coeffs':        \>\"", "<>", 
          RowBox[{"ToPython", "[", "coeffs", "]"}], "<>", 
          "\"\<,\n        'k_moduli':      \>\"", "<>", 
          RowBox[{"ToPython", "[", "kahlerModuli", "]"}], "<>", 
          "\"\<,\n        'ambient_dims':  \>\"", "<>", 
          RowBox[{"ToPython", "[", "dimPs", "]"}], " ", "<>", 
          "\"\<,\n        'precision':     \>\"", "<>", 
          RowBox[{"ToPython", "[", "precision", "]"}], " ", "<>", 
          "\"\<,\n        'vol_j_norm':    \>\"", "<>", 
          RowBox[{"ToPython", "[", "volJNorm", "]"}], " ", "<>", 
          "\"\<,\n        'selected_t': \>\"", "<>", 
          RowBox[{"ToPython", "[", "numParamsInPn", "]"}], " ", "<>", 
          "\"\<,\n        'point_file_path':\\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "pointsFile", "]"}], " ", "<>", 
          "\"\<\\\"\n       }\>\""}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{"session", ",", 
           RowBox[{"\"\<mcy.generate_points\>\"", "\[Rule]", "args"}]}], 
          "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";"}], ",", 
          "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"DeleteFile", "[", "pointsFile", "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{"res", ",", "session"}], "}"}], "]"}], ";"}], 
       "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GeneratePointsToric", "::", "usage"}], "=", 
   "\"\<GeneratePointsToric[PathToToricInfo,Options] generates points on the \
CY specified by the toric variety that has been analyzed by SAGE in \
PathToToricInfo. \\n* Input:\\n  - PathToToricInfo (str): Path where SAGE \
stored the info on the toirc CY.\\n* Options (run Options[GeneratePoints] to \
see default values):\\n  - KahlerModuli (list of floats): Kahler moduli \
\!\(\*SubscriptBox[\(t\), \(i\)]\) of the i'th ambient space factor (defaults \
to all 1)\\n  - Points (int): Number of points to generate (defaults to \
200,000)\\n  - Precision (int): WorkingPrecision to use when generating the \
points (defaults to 20)\\n  - Python (string): python executable to use \
(defaults to the one of $SETTINGSFILE)\\n  - Session (ExternalSessionObject): \
Python session with all dependencies loaded (if not provided, a new one will \
be generated)\\n  - Dir (string): Directory where points will be saved\\n  - \
Verbose (int): Verbose level (the higher, the more info is printed)\\n* \
Return:\\n  - res (object): Null if no error occured, otherwise a string with \
the error\\n  - session (ExternalSessionObject): The session object used in \
the coomputation (for potential faster future reuse) \\n* Example:\\n  - \
GeneratePoints[{\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(0\)], \
\(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(1\)], \
\(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(2\)], \
\(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(3\)], \
\(5\)]\)+\!\(\*SuperscriptBox[SubscriptBox[\(z\), \(4\)], \
\(5\)]\)},{4},\\\"Points\\\"\[Rule]200000,\\\"KahlerModuli\\\"\[Rule]{1},\\\"\
Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "GeneratePointsToric", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<KahlerModuli\>\"", "\[Rule]", 
      RowBox[{"{", "}"}]}], ",", 
     RowBox[{"\"\<Points\>\"", "\[Rule]", "200000"}], ",", 
     RowBox[{"\"\<Precision\>\"", "\[Rule]", "20"}], ",", 
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}], ",", 
     RowBox[{"\"\<Verbose\>\"", "\[Rule]", "3"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GeneratePointsToric", "[", 
     RowBox[{"PathToToricInfo_", ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "points", ",", "numPts", ",", "pointsFile", ",", 
        "verbose", ",", "session", ",", "outDir", ",", "res", ",", "startPos",
         ",", " ", "monomials", ",", " ", "coeffs", ",", "kahlerModuli", ",", 
        " ", "precision", ",", "loggerLevel", ",", "args", ",", "vars", ",", 
        "randomPoint", ",", "i", ",", "prev", ",", "curr", ",", "isSymmetric",
         ",", "pointsBatched", ",", "volJNorm", ",", "dimCY", ",", 
        "coefficients", ",", "dimPs", ",", "patchMasks", ",", "GLSMcharges", 
        ",", "sections"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"numPts", "=", 
         RowBox[{"OptionValue", "[", "\"\<Points\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"verbose", "=", 
         RowBox[{"OptionValue", "[", "\"\<Verbose\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"kahlerModuli", "=", 
         RowBox[{"OptionValue", "[", "\"\<KahlerModuli\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"precision", "=", 
         RowBox[{"OptionValue", "[", "\"\<Precision\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{"Start", " ", "session"}], ",", " ", 
          RowBox[{
          "load", " ", "in", " ", "prepare", " ", "data", " ", "file"}]}], 
         "*)"}], "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", 
          RowBox[{"python", ",", "session"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"loggerLevel", "=", "\"\<ERROR\>\""}], ";", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{"Taken", " ", "for", " ", "verbose"}], " ", "\[LessEqual]", 
          " ", "0"}], "*)"}], "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[Equal]", "1"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<WARNING\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[Equal]", "2"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<INFO\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[GreaterEqual]", "3"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<DEBUG\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"(*", 
         RowBox[{"Read", " ", "in", " ", "toric", " ", "data"}], "*)"}], 
        "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{"session", ",", 
           RowBox[{
           "\"\<import pickle;import numpy as np;pickle.load(open(\\\"\>\"", "<>",
             "PathToToricInfo", "<>", "\"\<\\\",'rb'))\>\""}]}], "]"}]}], ";",
         "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";", 
           RowBox[{"Return", "[", 
            RowBox[{"res", ",", "session"}], "]"}]}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"dimCY", "=", 
         RowBox[{"res", "[", 
          RowBox[{"[", "\"\<dim_cy\>\"", "]"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"monomials", "=", 
         RowBox[{"res", "[", 
          RowBox[{"[", "\"\<exp_aK\>\"", "]"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"coefficients", "=", 
         RowBox[{"res", "[", 
          RowBox[{"[", "\"\<coeff_aK\>\"", "]"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"patchMasks", "=", 
         RowBox[{"res", "[", 
          RowBox[{"[", "\"\<patch_masks\>\"", "]"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"GLSMcharges", "=", 
         RowBox[{"res", "[", 
          RowBox[{"[", "\"\<glsm_charges\>\"", "]"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"sections", "=", 
         RowBox[{"res", "[", 
          RowBox[{"[", "\"\<exps_sections\>\"", "]"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"volJNorm", "=", 
         RowBox[{"res", "[", "\"\<vol_j_norm\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"dimPs", "=", 
         RowBox[{"{", 
          RowBox[{"Length", "[", 
           RowBox[{"monomials", "[", 
            RowBox[{"[", "1", "]"}], "]"}], "]"}], "}"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"kahlerModuli", " ", "=", " ", 
         RowBox[{"If", "[", 
          RowBox[{
           RowBox[{"kahlerModuli", "\[Equal]", 
            RowBox[{"{", "}"}]}], ",", " ", 
           RowBox[{"Table", "[", 
            RowBox[{"1", ",", 
             RowBox[{"{", 
              RowBox[{"i", ",", 
               RowBox[{"Length", "[", "dimPs", "]"}]}], "}"}]}], "]"}], ",", 
           "kahlerModuli"}], "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"pointsFile", "=", 
         RowBox[{"FileNameJoin", "[", 
          RowBox[{"{", 
           RowBox[{"outDir", ",", "\"\<points.pickle\>\""}], "}"}], "]"}]}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"args", "=", 
         RowBox[{
         "\"\<{\n        'toric_file_path': \\\"\>\"", "<>", 
          "PathToToricInfo", "<>", 
          "\"\<\\\",\n        'outdir':        \\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
          "\"\<\\\",\n        'logger_level':  logging.\>\"", "<>", 
          "loggerLevel", "<>", "\"\<,\n        'num_pts':       \>\"", "<>", 
          RowBox[{"ToPython", "[", "numPts", "]"}], "<>", 
          "\"\<,\n        'dim_cy':        \>\"", "<>", 
          RowBox[{"ToPython", "[", "dimCY", "]"}], "<>", 
          "\"\<,\n        'monomials':     \>\"", "<>", 
          RowBox[{"ToPython", "[", "monomials", "]"}], "<>", 
          "\"\<,\n        'coeffs':        \>\"", "<>", 
          RowBox[{"ToPython", "[", "coefficients", "]"}], "<>", 
          "\"\<,\n        'k_moduli':      \>\"", "<>", 
          RowBox[{"ToPython", "[", "kahlerModuli", "]"}], "<>", 
          "\"\<,\n        'ambient_dims':  \>\"", "<>", 
          RowBox[{"ToPython", "[", "dimPs", "]"}], "<>", 
          "\"\<,\n        'sections':      \>\"", "<>", 
          RowBox[{"ToPython", "[", "sections", "]"}], "<>", 
          "\"\<,\n        'patch_masks':   \>\"", "<>", 
          RowBox[{"ToPython", "[", "patchMasks", "]"}], "<>", 
          "\"\<,\n        'glsm_charges':  \>\"", "<>", 
          RowBox[{"ToPython", "[", "GLSMcharges", "]"}], "<>", 
          "\"\<,\n        'precision':     \>\"", "<>", 
          RowBox[{"ToPython", "[", "precision", "]"}], " ", "<>", 
          "\"\<,\n        'vol_j_norm':    \>\"", "<>", 
          RowBox[{"ToPython", "[", "volJNorm", "]"}], " ", "<>", 
          "\"\<,\n        'verbose':       \>\"", "<>", 
          RowBox[{"ToPython", "[", "verbose", "]"}], " ", "<>", 
          "\"\<,\n        'point_file_path':\\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "pointsFile", "]"}], "<>", 
          "\"\<\\\"\n       }\>\""}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"Print", "[", 
         RowBox[{
         "\"\<Generating \>\"", ",", "numPts", ",", "\"\< points...\>\""}], 
         "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"points", "=", 
         RowBox[{"GenerateToricPointsM", "[", 
          RowBox[{
          "numPts", ",", "dimCY", ",", "coefficients", ",", "monomials", ",", 
           "sections", ",", "patchMasks", ",", "GLSMcharges", ",", 
           "precision", ",", "verbose"}], "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"Print", "[", 
         RowBox[{"\"\<Writing points to \>\"", ",", "pointsFile"}], "]"}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"!", 
           RowBox[{"DirectoryQ", "[", "outDir", "]"}]}], ",", 
          RowBox[{"CreateDirectory", "[", "outDir", "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"(*", " ", 
         RowBox[{
          RowBox[{"batch", " ", "the", " ", "conversion"}], ",", " ", 
          RowBox[{
          "otherwise", " ", "there", " ", "is", " ", "an", " ", "error", " ", 
           "for", " ", "long", " ", "points"}]}], "*)"}], 
        "\[IndentingNewLine]", 
        RowBox[{"pointsBatched", "=", 
         RowBox[{"Partition", "[", 
          RowBox[{"points", ",", 
           RowBox[{"UpTo", "[", "1000", "]"}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"ExternalEvaluate", "[", 
         RowBox[{"session", ",", "\"\<generated_pts=[];\>\""}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"For", "[", 
         RowBox[{
          RowBox[{"i", "=", "1"}], ",", 
          RowBox[{"i", "\[LessEqual]", 
           RowBox[{"Length", "[", "pointsBatched", "]"}]}], ",", 
          RowBox[{"i", "++"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"res", "=", 
            RowBox[{"ExternalEvaluate", "[", 
             RowBox[{"session", ",", 
              RowBox[{"\"\<generated_pts +=\>\"", "<>", 
               RowBox[{"ToPython", "[", 
                RowBox[{"pointsBatched", "[", 
                 RowBox[{"[", "i", "]"}], "]"}], "]"}], "<>", "\"\<;\>\""}]}],
              "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FailureQ", "[", "res", "]"}], ",", 
             RowBox[{
              RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
              
              RowBox[{"Print", "[", "res", "]"}], ";", 
              RowBox[{"Break", "[", "]"}], ";"}]}], "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{"session", ",", 
           RowBox[{
           "\"\<import pickle;import numpy as \
np;pickle.dump(np.array(generated_pts), open(\\\"\>\"", "<>", "pointsFile", 
            "<>", "\"\<\\\",'wb'));\>\""}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"ExternalEvaluate", "[", 
         RowBox[{"session", ",", "\"\<generated_pts=[];\>\""}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{"session", ",", 
           RowBox[{"\"\<mcy.generate_points_toric\>\"", "\[Rule]", "args"}]}],
           "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";"}], ",", 
          "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"DeleteFile", "[", "pointsFile", "]"}], ";"}]}], 
         "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{"res", ",", "session"}], "}"}], "]"}], ";"}], 
       "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"TrainNN", "::", "usage"}], "=", 
   "\"\<TrainNN[Options] trains a NN to approximate the CY metric.\\n* \
Options (run Options[TrainNN] to see default values):\\n  - Model (string): \
Choices are:\\n     - PhiFS: The NN learns a scalar function \[Phi] s.t. \!\(\
\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)+\
\[PartialD]\!\(\*OverscriptBox[\(\[PartialD]\), \(_\)]\)\[Phi]\\n     - \
PhiFSToric: Like PhiFS, but for CYs built from 4D toric varieties (need to \
specify location of toric info as generated from SAGE in Option \
ToricDataPath).\\n     - MultFS: The NN learns a matrix \
\!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\
\!\(\*SubscriptBox[\(g\), \(FS\)]\) * (id + \!\(\*SubscriptBox[\(g\), \(NN\)]\
\)) where id is the identity matrix and \\\"*\\\" is component-wise \
multiplication \\n     - MatrixMultFS: The NN learns a matrix \
\!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\
\!\(\*SubscriptBox[\(g\), \(FS\)]\)(id + \!\(\*SubscriptBox[\(g\), \(NN\)]\)) \
where id is the identity matrix and the multiplication is standard matrix \
multiplication \\n     - AddFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\
\), \(NN\)]\) s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\
\), \(FS\)]\) + \!\(\*SubscriptBox[\(g\), \(NN\)]\) \\n     - Free: The NN \
learns the CY metric directly, i.e. \!\(\*SubscriptBox[\(g\), \
\(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(NN\)]\)  \\n     - MatrixMultFSToric: \
Like MatrixMultFS, but for CYs built from 4D toric varieties (need to specify \
location of toric info as generated from SAGE in Option ToricDataPath).\\n  - \
EvaluateModel (bool): If True, computes different quantities for the test \
set, such as the \[Sigma]-measure, Kahler Loss, Transition Loss, change of \
Kahler class, and Ricci Scalar. Especially the Kahler and Ricci evaluations \
can be very slow, so only activate if feasible performance-wise  \\n  - \
HiddenLayers (list of ints): Number of hidden nodes in each hidden layer \\n  \
- ActivationFunctions (list of strings): Tensorflow activation function to \
use\\n  - Epochs (int): Number of training epochs\\n  - BatchSize (int): \
Batch size for training\\n  - Kappa (float): Value for kappa=\[Integral]|\
\[CapitalOmega]|^2/\[Integral] J^3; if 0 is passed, it will be computed \
automatically.\\n  - Alphas (list of floats): Relative weight of losses \
(Sigma, Kahler, Transition, Ricci, volK); at the moment, the value for Ricci \
is ignored since we do not train against the Ricci loss\\n  - Python \
(string): python executable to use (defaults to the one of $SETTINGSFILE)\\n  \
- Session (ExternalSessionObject): Python session with all dependencies \
loaded (if not provided, a new one will be generated)\\n  - Dir (string): \
Directory where the training points will be read from and the trained NN will \
be saved to\\n  - Verbose (int): Verbose level (the higher, the more info is \
printed)\\n* Return:\\n  - res (object): Null if no error occured, otherwise \
a string with the error\\n  - session (ExternalSessionObject): The session \
object used in the coomputation (for potential faster future reuse) \\n* \
Example:\\n  - \
TrainNN[\\\"HiddenLayers\\\"\[Rule]{64,64,64},\\\"ActivationFunctions\\\"\
\[Rule]{\\\"gelu\\\",\\\"gelu\\\",\\\"gelu\\\"},\\\"Epochs\\\"\[Rule]20,\\\"\
BatchSize\\\"\[Rule]64,\\\"Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "TrainNN", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<Model\>\"", "\[Rule]", "\"\<MultFS\>\""}], ",", 
     RowBox[{"\"\<EvaluateModel\>\"", "\[Rule]", "False"}], ",", 
     RowBox[{"\"\<HiddenLayers\>\"", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"64", ",", "64", ",", "64"}], "}"}]}], ",", 
     RowBox[{"\"\<ActivationFunctions\>\"", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"\"\<gelu\>\"", ",", "\"\<gelu\>\"", ",", "\"\<gelu\>\""}], 
       "}"}]}], ",", 
     RowBox[{"\"\<Epochs\>\"", "\[Rule]", "20"}], ",", 
     RowBox[{"\"\<BatchSize\>\"", "\[Rule]", "64"}], ",", 
     RowBox[{"\"\<Kappa\>\"", "\[Rule]", "0."}], ",", 
     RowBox[{"\"\<Alphas\>\"", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"1.", ",", "1.", ",", "1.", ",", "1.", ",", "1."}], "}"}]}], 
     ",", 
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}], ",", 
     RowBox[{"\"\<ToricDataPath\>\"", "\[Rule]", "\"\<\>\""}], ",", 
     RowBox[{"\"\<Verbose\>\"", "\[Rule]", "3"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"TrainNN", "[", 
     RowBox[{"OptionsPattern", "[", "]"}], "]"}], ":=", "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "model", ",", "callbacks", ",", "nHiddens", ",", "acts",
         ",", "nEpochs", ",", "batchSize", ",", "kappa", ",", "alphas", ",", 
        "outDir", ",", "verbose", ",", "session", ",", "res", ",", " ", 
        "loggerLevel", ",", "args", ",", "validSession", ",", 
        "toricDataPath"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"model", "=", 
         RowBox[{"OptionValue", "[", "\"\<Model\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"callbacks", "=", 
         RowBox[{"OptionValue", "[", "\"\<EvaluateModel\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"nHiddens", "=", 
         RowBox[{"OptionValue", "[", "\"\<HiddenLayers\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"acts", "=", 
         RowBox[{"OptionValue", "[", "\"\<ActivationFunctions\>\"", "]"}]}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"nEpochs", "=", 
         RowBox[{"OptionValue", "[", "\"\<Epochs\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"batchSize", "=", 
         RowBox[{"OptionValue", "[", "\"\<BatchSize\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"kappa", "=", 
         RowBox[{"OptionValue", "[", "\"\<Kappa\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"alphas", "=", 
         RowBox[{"OptionValue", "[", "\"\<Alphas\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"verbose", "=", 
         RowBox[{"OptionValue", "[", "\"\<Verbose\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"toricDataPath", "=", 
         RowBox[{"OptionValue", "[", "\"\<ToricDataPath\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"toricDataPath", "\[Equal]", "\"\<\>\""}], "&&", 
           RowBox[{"(", 
            RowBox[{
             RowBox[{"model", "\[Equal]", "\"\<PhiFSToric\>\""}], "||", 
             RowBox[{"model", "\[Equal]", "\"\<MatrixMultFSToric\>\""}]}], 
            ")"}]}], ",", 
          RowBox[{
           RowBox[{
           "Print", "[", 
            "\"\<Need to specify path to toric info as generated by \
SAGE.\>\"", "]"}], ";", 
           RowBox[{"Return", "[", 
            RowBox[{"{", 
             RowBox[{
             "\"\<Need to specify path to toric info as generated by \
SAGE.\>\"", ",", "session"}], "}"}], "]"}]}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", 
          RowBox[{"python", ",", "session"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"loggerLevel", "=", "\"\<ERROR\>\""}], ";", 
        RowBox[{"(*", 
         RowBox[{
          RowBox[{"Taken", " ", "for", " ", "verbose"}], " ", "\[LessEqual]", 
          " ", "0"}], "*)"}], "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[Equal]", "1"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<WARNING\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[Equal]", "2"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<INFO\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[GreaterEqual]", "3"}], ",", " ", 
          RowBox[{"loggerLevel", "=", "\"\<DEBUG\>\""}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"args", "=", 
         RowBox[{"\"\<{\n        'outdir':        \\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
          "\"\<\\\",\n        'logger_level':  logging.\>\"", "<>", 
          "loggerLevel", "<>", "\"\<,\n        'model':         \\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "model", "]"}], "<>", 
          "\"\<\\\",\n        'callbacks':     \>\"", "<>", 
          RowBox[{"ToPython", "[", "callbacks", "]"}], "<>", 
          "\"\<,\n        'n_hiddens':     \>\"", "<>", 
          RowBox[{"ToPython", "[", "nHiddens", "]"}], "<>", 
          "\"\<,\n\t\t'acts':          \>\"", "<>", 
          RowBox[{"ToPython", "[", "acts", "]"}], "<>", 
          "\"\<,\n\t\t'n_epochs':      \>\"", "<>", 
          RowBox[{"ToPython", "[", "nEpochs", "]"}], "<>", 
          "\"\<,\n\t\t'batch_size':    \>\"", "<>", 
          RowBox[{"ToPython", "[", "batchSize", "]"}], "<>", 
          "\"\<,\n\t\t'kappa':         \>\"", "<>", 
          RowBox[{"ToPython", "[", "kappa", "]"}], "<>", 
          "\"\<,\n        'alphas':        \>\"", "<>", 
          RowBox[{"ToPython", "[", "alphas", "]"}], "<>", 
          "\"\<,\n        'toric_data_path':\\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "toricDataPath", "]"}], "<>", 
          "\"\<\\\"\n       }\>\""}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"verbose", "\[GreaterEqual]", "3"}], ",", 
          RowBox[{"Print", "[", "args", "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{"session", ",", 
           RowBox[{"\"\<mcy.train_NN\>\"", "\[Rule]", "args"}]}], "]"}]}], 
        ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";", 
           RowBox[{"Return", "[", 
            RowBox[{"res", ",", "session"}], "]"}]}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Print", "[", 
         RowBox[{"\"\<Writing training information to \>\"", "<>", 
          RowBox[{"FileNameJoin", "[", 
           RowBox[{"{", 
            RowBox[{"outDir", ",", "\"\<training_history_mathematica.m\>\""}],
             "}"}], "]"}]}], "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Export", "[", 
         RowBox[{
          RowBox[{"FileNameJoin", "[", 
           RowBox[{"{", 
            RowBox[{"outDir", ",", "\"\<trianing_history_mathematica.m\>\""}],
             "}"}], "]"}], ",", "res"}], "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{"res", ",", "session"}], "}"}], "]"}], ";"}], 
       "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetPoints", "::", "usage"}], "=", 
   "\"\<GetPoints[dataset,Options] gets the CY points generated by the point \
generator.\\n* Input:\\n  - dataset (string): \\\"train\\\" for training \
dataset, \\\"val\\\" for validation dataset, \\\"all\\\" for both training \
and validation dataset\\n* Options (run Options[GetPoints] to see default \
values):\\n  - Python (string): python executable to use (defaults to the one \
of $SETTINGSFILE)\\n  - Session (ExternalSessionObject): Python session with \
all dependencies loaded (if not provided, a new one will be generated)\\n  - \
Dir (string): Directory where points are saved\\n* Return:\\n  - res \
(object): points (as list of complex lists) if no error occured, otherwise a \
string with the error\\n  - session (ExternalSessionObject): The session \
object used in the coomputation (for potential faster future reuse) \\n* \
Example:\\n  - GetPoints[\\\"val\\\",\\\"Dir\\\"\[Rule]\\\"./test\\\"]\>\""}],
   ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "GetPoints", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}]}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GetPoints", "[", 
     RowBox[{
      RowBox[{"dataset_", ":", "\"\<all\>\""}], ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "session", ",", "outDir", ",", "res", ",", "lenPts"}], 
       "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", 
          RowBox[{"python", ",", "session"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"dataset", "\[Equal]", "\"\<all\>\""}], ",", 
          "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"res", "=", 
            RowBox[{"Join", "[", "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"ExternalEvaluate", "[", 
               RowBox[{"session", ",", 
                RowBox[{
                "\"\<import numpy as np;import os;data=np.load(os.path.join('\
\>\"", "<>", 
                 RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                 "\"\<', 'dataset.npz'));data['X_train']\>\""}]}], "]"}], ",", 
              RowBox[{"ExternalEvaluate", "[", 
               RowBox[{"session", ",", 
                RowBox[{
                "\"\<import numpy as np;import os;data=np.load(os.path.join('\
\>\"", "<>", 
                 RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                 "\"\<', 'dataset.npz'));data['X_val']\>\""}]}], "]"}]}], 
             "\[IndentingNewLine]", "]"}]}], ";"}], ",", 
          "\[IndentingNewLine]", 
          RowBox[{"res", "=", 
           RowBox[{"ExternalEvaluate", "[", 
            RowBox[{"session", ",", 
             RowBox[{
             "\"\<import numpy as np;import \
os;data=np.load(os.path.join('\>\"", "<>", 
              RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
              "\"\<', 'dataset.npz'));data['X_\>\"", "<>", "dataset", "<>", 
              "\"\<']\>\""}]}], "]"}]}]}], "\[IndentingNewLine]", "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";", 
           RowBox[{"Return", "[", 
            RowBox[{"{", 
             RowBox[{"res", ",", "session"}], "}"}], "]"}]}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"Length", "[", "res", "]"}], ">", "0"}], ",", 
          "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"lenPts", "=", 
            RowBox[{"Floor", "[", 
             RowBox[{
              RowBox[{"Length", "[", 
               RowBox[{"res", "[", 
                RowBox[{"[", "1", "]"}], "]"}], "]"}], "/", "2"}], "]"}]}], 
           ";", "\[IndentingNewLine]", 
           RowBox[{"res", "=", 
            RowBox[{
             RowBox[{"res", "[", 
              RowBox[{"[", 
               RowBox[{";;", ",", 
                RowBox[{"1", ";;", "lenPts"}]}], "]"}], "]"}], "+", 
             RowBox[{"I", "*", 
              RowBox[{"res", "[", 
               RowBox[{"[", 
                RowBox[{";;", ",", 
                 RowBox[{
                  RowBox[{"lenPts", "+", "1"}], ";;"}]}], "]"}], "]"}]}]}]}], 
           ";"}], "\[IndentingNewLine]", ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"res", "=", 
            RowBox[{"{", "}"}]}], ";"}]}], "\[IndentingNewLine]", "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{
           RowBox[{"Chop", "[", 
            RowBox[{"Normal", "[", "res", "]"}], "]"}], ",", "session"}], 
          "}"}], "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}],
   "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetFSWeights", "::", "usage"}], "=", 
   "\"\<GetFSWeights[dataset,Options] gets the weights of the CY points \
generated by the point generator for the Fubini-Study metric.\\n* Input:\\n  \
- dataset (string or list): \\\"train\\\" for training dataset, \\\"val\\\" \
for validation dataset, \\\"all\\\" for both training and validation dataset, \
or a list of points\\n* Options (run Options[GetFSWeights] to see default \
values):\\n  - Python (string): python executable to use (defaults to the one \
of $SETTINGSFILE)\\n  - Session (ExternalSessionObject): Python session with \
all dependencies loaded (if not provided, a new one will be generated)\\n  - \
Dir (string): Directory where weights are saved\\n  - Dir (string): Directory \
where CY information is saved\\n* Return:\\n  - res (object): weights (as \
list of floats) if no error occured, otherwise a string with the error\\n  - \
session (ExternalSessionObject): The session object used in the coomputation \
(for potential faster future reuse) \\n* Example:\\n  - \
GetFSWeights[\\\"val\\\",\\\"Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "GetFSWeights", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}]}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GetFSWeights", "[", 
     RowBox[{
      RowBox[{"dataset_", ":", "\"\<all\>\""}], ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "session", ",", "outDir", ",", "res", ",", "pts", ",", 
        "args"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", 
          RowBox[{"python", ",", "session"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"ListQ", "[", "dataset", "]"}], ",", "\[IndentingNewLine]", 
          
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Length", "[", "dataset", "]"}], "\[Equal]", "0"}], ",", 
             RowBox[{"Return", "[", 
              RowBox[{"{", 
               RowBox[{
                RowBox[{"{", "}"}], ",", "session"}], "}"}], "]"}]}], "]"}], 
           ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Length", "[", 
               RowBox[{"Dimensions", "[", "dataset", "]"}], "]"}], "==", 
              "1"}], ",", 
             RowBox[{"pts", "=", 
              RowBox[{"{", 
               RowBox[{"Join", "[", 
                RowBox[{
                 RowBox[{"Re", "[", "dataset", "]"}], ",", 
                 RowBox[{"Im", "[", 
                  RowBox[{"dataset", "[", 
                   RowBox[{"[", ";;", "]"}], "]"}], "]"}]}], "]"}], "}"}]}], 
             ",", 
             RowBox[{"pts", "=", 
              RowBox[{"Table", "[", 
               RowBox[{
                RowBox[{"Join", "[", 
                 RowBox[{
                  RowBox[{"Re", "[", 
                   RowBox[{"dataset", "[", 
                    RowBox[{"[", "i", "]"}], "]"}], "]"}], ",", 
                  RowBox[{"Im", "[", 
                   RowBox[{"dataset", "[", 
                    RowBox[{"[", "i", "]"}], "]"}], "]"}]}], "]"}], ",", 
                RowBox[{"{", 
                 RowBox[{"i", ",", 
                  RowBox[{"Length", "[", "dataset", "]"}]}], "}"}]}], 
               "]"}]}]}], "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"args", "=", 
            RowBox[{"\"\<{\n        'outdir':      \\\"\>\"", "<>", 
             RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
             "\"\<\\\",\n        'points':        \>\"", "<>", 
             RowBox[{"ToPython", "[", "pts", "]"}], "<>", 
             "\"\<\n       }\>\""}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"res", "=", 
            RowBox[{"ExternalEvaluate", "[", 
             RowBox[{"session", ",", 
              RowBox[{"\"\<mcy.get_weights\>\"", "\[Rule]", "args"}]}], 
             "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FailureQ", "[", "res", "]"}], ",", 
             RowBox[{
              RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
              
              RowBox[{"Print", "[", "res", "]"}], ";", 
              RowBox[{"Return", "[", 
               RowBox[{"{", 
                RowBox[{"res", ",", "session"}], "}"}], "]"}]}]}], "]"}], 
           ";"}], "\[IndentingNewLine]", ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"dataset", "\[Equal]", "\"\<all\>\""}], ",", 
             "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"res", "=", 
               RowBox[{"Join", "[", "\[IndentingNewLine]", 
                RowBox[{
                 RowBox[{"ExternalEvaluate", "[", 
                  RowBox[{"session", ",", 
                   RowBox[{
                   "\"\<import numpy as np;import \
os;data=np.load(os.path.join('\>\"", "<>", 
                    RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                    "\"\<', 'dataset.npz'));data['y_train']\>\""}]}], "]"}], 
                 ",", 
                 RowBox[{"ExternalEvaluate", "[", 
                  RowBox[{"session", ",", 
                   RowBox[{
                   "\"\<import numpy as np;import \
os;data=np.load(os.path.join('\>\"", "<>", 
                    RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                    "\"\<', 'dataset.npz'));data['y_val']\>\""}]}], "]"}]}], 
                "\[IndentingNewLine]", "]"}]}], ";"}], ",", 
             "\[IndentingNewLine]", 
             RowBox[{"res", "=", 
              RowBox[{"ExternalEvaluate", "[", 
               RowBox[{"session", ",", 
                RowBox[{
                "\"\<import numpy as np;import os;data=np.load(os.path.join('\
\>\"", "<>", 
                 RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                 "\"\<', 'dataset.npz'));data['y_\>\"", "<>", "dataset", 
                 "<>", "\"\<']\>\""}]}], "]"}]}]}], "\[IndentingNewLine]", 
            "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FailureQ", "[", "res", "]"}], ",", 
             RowBox[{
              RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
              
              RowBox[{"Print", "[", "res", "]"}], ";", 
              RowBox[{"Return", "[", 
               RowBox[{"{", 
                RowBox[{"res", ",", "session"}], "}"}], "]"}]}]}], "]"}], ";",
            "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Length", "[", "res", "]"}], ">", "0"}], ",", 
             "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"res", "=", 
               RowBox[{"res", "[", 
                RowBox[{"[", 
                 RowBox[{";;", ",", 
                  RowBox[{"-", "2"}]}], "]"}], "]"}]}], ";"}], 
             "\[IndentingNewLine]", ",", "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"res", "=", 
               RowBox[{"{", "}"}]}], ";"}]}], "\[IndentingNewLine]", "]"}], 
           ";"}]}], "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{
           RowBox[{"Normal", "[", "res", "]"}], ",", "session"}], "}"}], 
         "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetOmegaSquared", "::", "usage"}], "=", 
   "\"\<GetOmegaSquared[dataset,Options] gets \
(|\[CapitalOmega]|\!\(\*SuperscriptBox[\()\), \(2\)]\) of the CY generated by \
the point generator.\\n* Input:\\n  - dataset (string): \\\"train\\\" for \
training dataset, \\\"val\\\" for validation dataset, \\\"all\\\" for both \
training and validation dataset, or a list of points\\n* Options (run \
Options[GetOmegaSquared] to see default values):\\n  - Python (string): \
python executable to use (defaults to the one of $SETTINGSFILE)\\n  - Session \
(ExternalSessionObject): Python session with all dependencies loaded (if not \
provided, a new one will be generated)\\n  - Dir (string): Directory where \
Omegas are saved\\n* Return:\\n  - res (object): \
(|\[CapitalOmega]|\!\(\*SuperscriptBox[\()\), \(2\)]\) (as list of floats) if \
no error occured, otherwise a string with the error\\n  - session \
(ExternalSessionObject): The session object used in the coomputation (for \
potential faster future reuse) \\n* Example:\\n  - \
GetOmegaSquared[\\\"val\\\",\\\"Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"Options", "[", "GetOmegaSquared", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}]}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GetOmegaSquared", "[", 
     RowBox[{
      RowBox[{"dataset_", ":", "\"\<all\>\""}], ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "session", ",", "outDir", ",", "res", ",", "pts", ",", 
        "args"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", 
          RowBox[{"python", ",", "session"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"ListQ", "[", "dataset", "]"}], ",", "\[IndentingNewLine]", 
          
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Length", "[", "dataset", "]"}], "\[Equal]", "0"}], ",", 
             RowBox[{"Return", "[", 
              RowBox[{"{", 
               RowBox[{
                RowBox[{"{", "}"}], ",", "session"}], "}"}], "]"}]}], "]"}], 
           ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Length", "[", 
               RowBox[{"Dimensions", "[", "dataset", "]"}], "]"}], "==", 
              "1"}], ",", 
             RowBox[{"pts", "=", 
              RowBox[{"{", 
               RowBox[{"Join", "[", 
                RowBox[{
                 RowBox[{"Re", "[", "dataset", "]"}], ",", 
                 RowBox[{"Im", "[", 
                  RowBox[{"dataset", "[", 
                   RowBox[{"[", ";;", "]"}], "]"}], "]"}]}], "]"}], "}"}]}], 
             ",", 
             RowBox[{"pts", "=", 
              RowBox[{"Table", "[", 
               RowBox[{
                RowBox[{"Join", "[", 
                 RowBox[{
                  RowBox[{"Re", "[", 
                   RowBox[{"dataset", "[", 
                    RowBox[{"[", "i", "]"}], "]"}], "]"}], ",", 
                  RowBox[{"Im", "[", 
                   RowBox[{"dataset", "[", 
                    RowBox[{"[", "i", "]"}], "]"}], "]"}]}], "]"}], ",", 
                RowBox[{"{", 
                 RowBox[{"i", ",", 
                  RowBox[{"Length", "[", "dataset", "]"}]}], "}"}]}], 
               "]"}]}]}], "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"args", "=", 
            RowBox[{"\"\<{\n        'outdir':      \\\"\>\"", "<>", 
             RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
             "\"\<\\\",\n        'points':        \>\"", "<>", 
             RowBox[{"ToPython", "[", "pts", "]"}], "<>", 
             "\"\<\n       }\>\""}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"res", "=", 
            RowBox[{"ExternalEvaluate", "[", 
             RowBox[{"session", ",", 
              RowBox[{"\"\<mcy.get_omegas\>\"", "\[Rule]", "args"}]}], 
             "]"}]}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FailureQ", "[", "res", "]"}], ",", 
             RowBox[{
              RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
              
              RowBox[{"Print", "[", "res", "]"}], ";", 
              RowBox[{"Return", "[", 
               RowBox[{"{", 
                RowBox[{"res", ",", "session"}], "}"}], "]"}]}]}], "]"}], 
           ";"}], "\[IndentingNewLine]", ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"dataset", "\[Equal]", "\"\<all\>\""}], ",", 
             "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"res", "=", 
               RowBox[{"Join", "[", "\[IndentingNewLine]", 
                RowBox[{
                 RowBox[{"ExternalEvaluate", "[", 
                  RowBox[{"session", ",", 
                   RowBox[{
                   "\"\<import numpy as np;import \
os;data=np.load(os.path.join('\>\"", "<>", 
                    RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                    "\"\<', 'dataset.npz'));data['y_train']\>\""}]}], "]"}], 
                 ",", 
                 RowBox[{"ExternalEvaluate", "[", 
                  RowBox[{"session", ",", 
                   RowBox[{
                   "\"\<import numpy as np;import \
os;data=np.load(os.path.join('\>\"", "<>", 
                    RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                    "\"\<', 'dataset.npz'));data['y_val']\>\""}]}], "]"}]}], 
                "\[IndentingNewLine]", "]"}]}], ";"}], ",", 
             "\[IndentingNewLine]", 
             RowBox[{"res", "=", 
              RowBox[{"ExternalEvaluate", "[", 
               RowBox[{"session", ",", 
                RowBox[{
                "\"\<import numpy as np;import os;data=np.load(os.path.join('\
\>\"", "<>", 
                 RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
                 "\"\<', 'dataset.npz'));data['y_\>\"", "<>", "dataset", 
                 "<>", "\"\<']\>\""}]}], "]"}]}]}], "\[IndentingNewLine]", 
            "]"}], ";", "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{"FailureQ", "[", "res", "]"}], ",", 
             RowBox[{
              RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
              
              RowBox[{"Print", "[", "res", "]"}], ";", 
              RowBox[{"Return", "[", 
               RowBox[{"{", 
                RowBox[{"res", ",", "session"}], "}"}], "]"}]}]}], "]"}], ";",
            "\[IndentingNewLine]", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Length", "[", "res", "]"}], ">", "0"}], ",", 
             "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"res", "=", 
               RowBox[{"res", "[", 
                RowBox[{"[", 
                 RowBox[{";;", ",", 
                  RowBox[{"-", "1"}]}], "]"}], "]"}]}], ";"}], 
             "\[IndentingNewLine]", ",", "\[IndentingNewLine]", 
             RowBox[{
              RowBox[{"res", "=", 
               RowBox[{"{", "}"}]}], ";"}]}], "\[IndentingNewLine]", "]"}], 
           ";"}]}], "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{
           RowBox[{"Re", "[", 
            RowBox[{"Normal", "[", "res", "]"}], "]"}], ",", "session"}], 
          "}"}], "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}],
   "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetCYWeights", "::", "usage"}], "=", 
   "\"\<GetCYWeights[dataset,Options] gets the weights of the CY points for \
the CY metric.\\n* Input:\\n  - dataset (string): \\\"train\\\" for training \
dataset, \\\"val\\\" for validation dataset, \\\"all\\\" for both training \
and validation dataset, or a list of points\\n* Options (run \
Options[GetCYWeights] to see default values):\\n  - Python (string): python \
executable to use (defaults to the one of $SETTINGSFILE)\\n  - Session \
(ExternalSessionObject): Python session with all dependencies loaded (if not \
provided, a new one will be generated)\\n  - Model (string): Choices are:\\n  \
   - PhiFS: The NN learns a scalar function \[Phi] s.t. \
\!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)+\
\[PartialD]\!\(\*OverscriptBox[\(\[PartialD]\), \(_\)]\)\[Phi]\\n     - \
MultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \
\!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) * (id \
+ \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and \\\
\"*\\\" is component-wise multiplication \\n     - MatrixMultFS: The NN \
learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \
\!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)(id + \
\!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and the \
multiplication is standard matrix multiplication \\n     - AddFS: The NN \
learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \
\!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) + \
\!\(\*SubscriptBox[\(g\), \(NN\)]\) \\n     - Free: The NN learns the CY \
metric directly, i.e. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\
\(g\), \(NN\)]\)\\n  - Kappa (float): Value for kappa=\[Integral]|\
\[CapitalOmega]|^2/\[Integral] J^3; if 0 is passed, it will be computed \
automatically.\\n  - Dir (string): Directory where weights and the trained NN \
are saved\\n* Return:\\n  - res (object): weights (as list of floats) if no \
error occured, otherwise a string with the error\\n  - session \
(ExternalSessionObject): The session object used in the coomputation (for \
potential faster future reuse) \\n* Example:\\n  - \
GetCYWeights[\\\"val\\\",\\\"Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], ";", 
  RowBox[{
   RowBox[{"Options", "[", "GetCYWeights", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Model\>\"", "\[Rule]", "\"\<MultFS\>\""}], ",", 
     RowBox[{"\"\<Kappa\>\"", "\[Rule]", "0."}], ",", 
     RowBox[{"\"\<VolJNorm\>\"", "\[Rule]", "1."}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}], ",", 
     RowBox[{"\"\<DimX\>\"", "\[Rule]", "3"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GetCYWeights", "[", 
     RowBox[{
      RowBox[{"dataset_", ":", "\"\<all\>\""}], ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "outDir", ",", "session", ",", "res", ",", "pts", ",", 
        "omegaSquared", ",", "gs", ",", "dets", ",", "dimX", ",", "tmp", ",", 
        "i", ",", "model", ",", "kappa", ",", "volJNorm"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"kappa", "=", 
         RowBox[{"OptionValue", "[", "\"\<Kappa\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"model", "=", 
         RowBox[{"OptionValue", "[", "\"\<Model\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"dimX", "=", 
         RowBox[{"OptionValue", "[", "\"\<DimX\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"volJNorm", "=", 
         RowBox[{"OptionValue", "[", "\"\<VolJNorm\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"GetSession", "[", 
          RowBox[{"python", ",", "session"}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", " ", "[", 
         RowBox[{
          RowBox[{"ListQ", "[", "dataset", "]"}], ",", "\[IndentingNewLine]", 
          
          RowBox[{
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{"Length", "[", 
               RowBox[{"Dimensions", "[", "dataset", "]"}], "]"}], "==", 
              "1"}], ",", 
             RowBox[{"pts", "=", 
              RowBox[{"{", "dataset", "}"}]}], ",", 
             RowBox[{"pts", "=", "dataset"}]}], "]"}], ";"}], 
          "\[IndentingNewLine]", ",", "\[IndentingNewLine]", 
          RowBox[{
           RowBox[{
            RowBox[{"{", 
             RowBox[{"pts", ",", "tmp"}], "}"}], "=", 
            RowBox[{"GetPoints", "[", 
             RowBox[{"dataset", ",", 
              RowBox[{"\"\<Python\>\"", "\[Rule]", "python"}], ",", 
              RowBox[{"\"\<Session\>\"", "\[Rule]", "session"}], ",", 
              RowBox[{"\"\<Dir\>\"", "\[Rule]", "outDir"}]}], "]"}]}], 
           ";"}]}], "\[IndentingNewLine]", "]"}], ";", "\[IndentingNewLine]", 
        
        RowBox[{
         RowBox[{"{", 
          RowBox[{"omegaSquared", ",", "tmp"}], "}"}], "=", 
         RowBox[{"GetOmegaSquared", "[", 
          RowBox[{"dataset", ",", 
           RowBox[{"\"\<Python\>\"", "\[Rule]", "python"}], ",", 
           RowBox[{"\"\<Session\>\"", "\[Rule]", "session"}], ",", 
           RowBox[{"\"\<Dir\>\"", "\[Rule]", "outDir"}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{
         RowBox[{"{", 
          RowBox[{"gs", ",", "tmp"}], "}"}], "=", 
         RowBox[{"CYMetric", "[", 
          RowBox[{"pts", ",", 
           RowBox[{"\"\<Python\>\"", "\[Rule]", "python"}], ",", 
           RowBox[{"\"\<Session\>\"", "\[Rule]", "session"}], ",", 
           RowBox[{"\"\<Model\>\"", "\[Rule]", "model"}], ",", 
           RowBox[{"\"\<Kappa\>\"", "\[Rule]", "kappa"}], ",", 
           RowBox[{"\"\<Dir\>\"", "\[Rule]", "outDir"}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"dets", "=", 
         RowBox[{"Table", "[", 
          RowBox[{
           RowBox[{"Det", "[", 
            RowBox[{"gs", "[", 
             RowBox[{"[", 
              RowBox[{"i", ",", "2"}], "]"}], "]"}], "]"}], ",", 
           RowBox[{"{", 
            RowBox[{"i", ",", 
             RowBox[{"Length", "[", "gs", "]"}]}], "}"}]}], "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"dets", "=", 
         RowBox[{"dets", " ", "/", "volJNorm"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"omegaSquared", "/", "dets"}]}], ";", "\[IndentingNewLine]", 
        
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{
           RowBox[{"Re", "[", 
            RowBox[{"Normal", "[", "res", "]"}], "]"}], ",", "session"}], 
          "}"}], "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}],
   "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"GetPullbacks", "::", "usage"}], "=", 
   "\"\<GetPullbacks[points,Options] computes the pullbacks from the ambient \
space coordinates to the CY at the given points. You need to call \
GeneratePoints[] first. The pullback will work for any point (n ot just the \
generated ones), but GeneratePoints[] computes several quantities (such as \
derivatives) needed for the pullback matrix. \\n* Input:\\n  - points (list): \
list of list of complex numbers that specify points on the CY\\n* Options \
(run Options[GetPullbacks] to see default values):\\n  - Python (string): \
python executable to use (defaults to the one of $SETTINGSFILE)\\n  - Session \
(ExternalSessionObject): Python session with all dependencies loaded (if not \
provided, a new one will be generated)\\n  - Dir (string): Directory where \
the point generator is saved.\\n* Return:\\n  - res (object): pullbacks (as \
list of float matrices) if no error occured, otherwise a string with the \
error\\n  - session (ExternalSessionObject): The session object used in the \
coomputation (for potential faster future reuse) \\n* Example:\\n  - \
GetPullbacks[{{1,2,3,4,5}},\\\"Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], ";", 
  RowBox[{
   RowBox[{"Options", "[", "GetPullbacks", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}]}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{
    RowBox[{"GetPullbacks", "[", 
     RowBox[{"points_", ",", 
      RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
    "\[IndentingNewLine]", 
    RowBox[{"Module", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{
       "python", ",", "outDir", ",", "session", ",", "res", ",", "args", ",", 
        "pts", ",", "kappa", ",", "model"}], "}"}], ",", 
      RowBox[{"(", "\[IndentingNewLine]", 
       RowBox[{
        RowBox[{"python", "=", 
         RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"session", "=", 
         RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"outDir", "=", 
         RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"session", "===", "Null"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
            "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
        "\[IndentingNewLine]", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"Length", "[", "points", "]"}], "\[Equal]", "0"}], ",", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{
             RowBox[{"{", "}"}], ",", "session"}], "}"}], "]"}]}], "]"}], ";",
         "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"Length", "[", 
            RowBox[{"Dimensions", "[", "points", "]"}], "]"}], "==", "1"}], 
          ",", 
          RowBox[{"pts", "=", 
           RowBox[{"{", 
            RowBox[{"Join", "[", 
             RowBox[{
              RowBox[{"Re", "[", "points", "]"}], ",", 
              RowBox[{"Im", "[", 
               RowBox[{"points", "[", 
                RowBox[{"[", ";;", "]"}], "]"}], "]"}]}], "]"}], "}"}]}], ",", 
          RowBox[{"pts", "=", 
           RowBox[{"Table", "[", 
            RowBox[{
             RowBox[{"Join", "[", 
              RowBox[{
               RowBox[{"Re", "[", 
                RowBox[{"points", "[", 
                 RowBox[{"[", "i", "]"}], "]"}], "]"}], ",", 
               RowBox[{"Im", "[", 
                RowBox[{"points", "[", 
                 RowBox[{"[", "i", "]"}], "]"}], "]"}]}], "]"}], ",", 
             RowBox[{"{", 
              RowBox[{"i", ",", 
               RowBox[{"Length", "[", "points", "]"}]}], "}"}]}], "]"}]}]}], 
         "]"}], ";", "\[IndentingNewLine]", 
        RowBox[{"args", "=", 
         RowBox[{"\"\<{\n        'outdir':      \\\"\>\"", "<>", 
          RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
          "\"\<\\\",\n        'points':        \>\"", "<>", 
          RowBox[{"ToPython", "[", "pts", "]"}], "<>", 
          "\"\<\n       }\>\""}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"res", "=", 
         RowBox[{"ExternalEvaluate", "[", 
          RowBox[{"session", ",", 
           RowBox[{"\"\<mcy.get_pullbacks\>\"", "\[Rule]", "args"}]}], 
          "]"}]}], ";", "\[IndentingNewLine]", 
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{"FailureQ", "[", "res", "]"}], ",", 
          RowBox[{
           RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
           RowBox[{"Print", "[", "res", "]"}], ";", 
           RowBox[{"Return", "[", 
            RowBox[{"{", 
             RowBox[{"res", ",", "session"}], "}"}], "]"}]}]}], "]"}], ";", 
        "\[IndentingNewLine]", 
        RowBox[{"Return", "[", 
         RowBox[{"{", 
          RowBox[{
           RowBox[{"Normal", "[", "res", "]"}], ",", "session"}], "}"}], 
         "]"}], ";"}], "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}], 
  "\[IndentingNewLine]"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"CYMetric", "::", "usage"}], "=", 
   "\"\<CYMetric[points,Options] computes the CY metric at the given \
points.\\n* Input:\\n  - points (list): list of list of complex numbers that \
specify points on the CY\\n* Options (run Options[CYMetric] to see default \
values):\\n  - Python (string): python executable to use (defaults to the one \
of $SETTINGSFILE)\\n  - Session (ExternalSessionObject): Python session with \
all dependencies loaded (if not provided, a new one will be generated)\\n  - \
Model (string): Choices are:\\n     - PhiFS: The NN learns a scalar function \
\[Phi] s.t. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \
\(FS\)]\)+\[PartialD]\!\(\*OverscriptBox[\(\[PartialD]\), \(_\)]\)\[Phi]\\n   \
  - MultFS: The NN learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \
\!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) * (id \
+ \!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and \\\
\"*\\\" is component-wise multiplication \\n     - MatrixMultFS: The NN \
learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \
\!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\)(id + \
\!\(\*SubscriptBox[\(g\), \(NN\)]\)) where id is the identity matrix and the \
multiplication is standard matrix multiplication \\n     - AddFS: The NN \
learns a matrix \!\(\*SubscriptBox[\(g\), \(NN\)]\) s.t. \
\!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\(g\), \(FS\)]\) + \
\!\(\*SubscriptBox[\(g\), \(NN\)]\) \\n     - Free: The NN learns the CY \
metric directly, i.e. \!\(\*SubscriptBox[\(g\), \(CY\)]\)=\!\(\*SubscriptBox[\
\(g\), \(NN\)]\)\\n  - Kappa (float): Value for kappa=\[Integral]|\
\[CapitalOmega]|^2/\[Integral] J^3; if 0 is passed, it will be computed \
automatically.\\n  - Dir (string): Directory where weights and the trained NN \
are saved\\n* Return:\\n  - res (object): weights (as list of floats) if no \
error occured, otherwise a string with the error\\n  - session \
(ExternalSessionObject): The session object used in the coomputation (for \
potential faster future reuse) \\n* Example:\\n  - \
CYMetric[{{1,2,3,4,5}},\\\"Dir\\\"\[Rule]\\\"./test\\\"]\>\""}], ";", 
  RowBox[{
   RowBox[{"Options", "[", "CYMetric", "]"}], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"\"\<Python\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Session\>\"", "\[Rule]", "Null"}], ",", 
     RowBox[{"\"\<Model\>\"", "\[Rule]", "\"\<MultFS\>\""}], ",", 
     RowBox[{"\"\<Kappa\>\"", "\[Rule]", "0."}], ",", 
     RowBox[{"\"\<Dir\>\"", "\[Rule]", 
      RowBox[{"FileNameJoin", "[", 
       RowBox[{"{", 
        RowBox[{
         RowBox[{"NotebookDirectory", "[", "]"}], ",", "\"\<test\>\""}], 
        "}"}], "]"}]}]}], "}"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   RowBox[{"CYMetric", "[", 
    RowBox[{"points_", ",", 
     RowBox[{"OptionsPattern", "[", "]"}]}], "]"}], ":=", 
   "\[IndentingNewLine]", 
   RowBox[{"Module", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{
      "python", ",", "outDir", ",", "session", ",", "res", ",", "args", ",", 
       "pts", ",", "kappa", ",", "model"}], "}"}], ",", 
     RowBox[{"(", "\[IndentingNewLine]", 
      RowBox[{
       RowBox[{"python", "=", 
        RowBox[{"OptionValue", "[", "\"\<Python\>\"", "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"session", "=", 
        RowBox[{"OptionValue", "[", "\"\<Session\>\"", "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"outDir", "=", 
        RowBox[{"OptionValue", "[", "\"\<Dir\>\"", "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"kappa", "=", 
        RowBox[{"OptionValue", "[", "\"\<Kappa\>\"", "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"model", "=", 
        RowBox[{"OptionValue", "[", "\"\<Model\>\"", "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"session", "=", 
        RowBox[{"GetSession", "[", 
         RowBox[{"python", ",", "session"}], "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"session", "===", "Null"}], ",", 
         RowBox[{"Return", "[", 
          RowBox[{"{", 
           RowBox[{
           "\"\<Could not start a Python Kernel with all dependencies \
installed.\>\"", ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
       "\[IndentingNewLine]", "\[IndentingNewLine]", 
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{
          RowBox[{"Length", "[", "points", "]"}], "\[Equal]", "0"}], ",", 
         RowBox[{"Return", "[", 
          RowBox[{"{", 
           RowBox[{
            RowBox[{"{", "}"}], ",", "session"}], "}"}], "]"}]}], "]"}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{
          RowBox[{"Length", "[", 
           RowBox[{"Dimensions", "[", "points", "]"}], "]"}], "==", "1"}], 
         ",", 
         RowBox[{"pts", "=", 
          RowBox[{"{", 
           RowBox[{"Join", "[", 
            RowBox[{
             RowBox[{"Re", "[", "points", "]"}], ",", 
             RowBox[{"Im", "[", 
              RowBox[{"points", "[", 
               RowBox[{"[", ";;", "]"}], "]"}], "]"}]}], "]"}], "}"}]}], ",", 
         
         RowBox[{"pts", "=", 
          RowBox[{"Table", "[", 
           RowBox[{
            RowBox[{"Join", "[", 
             RowBox[{
              RowBox[{"Re", "[", 
               RowBox[{"points", "[", 
                RowBox[{"[", "i", "]"}], "]"}], "]"}], ",", 
              RowBox[{"Im", "[", 
               RowBox[{"points", "[", 
                RowBox[{"[", "i", "]"}], "]"}], "]"}]}], "]"}], ",", 
            RowBox[{"{", 
             RowBox[{"i", ",", 
              RowBox[{"Length", "[", "points", "]"}]}], "}"}]}], "]"}]}]}], 
        "]"}], ";", "\[IndentingNewLine]", 
       RowBox[{"args", "=", 
        RowBox[{"\"\<{\n        'outdir':      \\\"\>\"", "<>", 
         RowBox[{"ToPython", "[", "outDir", "]"}], "<>", 
         "\"\<\\\",\n        'points':        \>\"", "<>", 
         RowBox[{"ToPython", "[", "pts", "]"}], "<>", 
         "\"\<,\n        'kappa':        \>\"", "<>", 
         RowBox[{"ToPython", "[", "kappa", "]"}], "<>", 
         "\"\<,\n        'model':        \\\"\>\"", "<>", 
         RowBox[{"ToPython", "[", "model", "]"}], "<>", 
         "\"\<\\\"\n       }\>\""}]}], ";", "\[IndentingNewLine]", 
       RowBox[{"res", "=", 
        RowBox[{"ExternalEvaluate", "[", 
         RowBox[{"session", ",", 
          RowBox[{"\"\<mcy.get_g\>\"", "\[Rule]", "args"}]}], "]"}]}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"If", "[", 
        RowBox[{
         RowBox[{"FailureQ", "[", "res", "]"}], ",", 
         RowBox[{
          RowBox[{"Print", "[", "\"\<An error occurred.\>\"", "]"}], ";", 
          RowBox[{"Print", "[", "res", "]"}], ";", 
          RowBox[{"Return", "[", 
           RowBox[{"{", 
            RowBox[{"res", ",", "session"}], "}"}], "]"}]}]}], "]"}], ";", 
       "\[IndentingNewLine]", 
       RowBox[{"Return", "[", 
        RowBox[{"{", 
         RowBox[{
          RowBox[{"Normal", "[", "res", "]"}], ",", "session"}], "}"}], "]"}],
        ";"}], "\[IndentingNewLine]", ")"}]}], "]"}]}], ";"}]}], "Input",
 InitializationCell->True,
 CellChangeTimes->{{3.82515366505571*^9, 3.825153734517518*^9}, {
   3.825159797843381*^9, 3.825160042783255*^9}, {3.82516083402944*^9, 
   3.825160866219452*^9}, {3.825160900356093*^9, 3.825161010209597*^9}, {
   3.825161050052167*^9, 3.8251612070126553`*^9}, 3.825161243895195*^9, {
   3.825161307846434*^9, 3.8251613877564*^9}, {3.825161473558344*^9, 
   3.825161579761334*^9}, {3.825161690744089*^9, 3.8251617018162413`*^9}, {
   3.825161787144664*^9, 3.825161900604589*^9}, {3.825161955681176*^9, 
   3.82516201782655*^9}, {3.825162227011751*^9, 3.825162231386503*^9}, {
   3.825162264533643*^9, 3.825162269619087*^9}, {3.82523411279115*^9, 
   3.8252342758274927`*^9}, {3.825234306251885*^9, 3.825234317957961*^9}, {
   3.8252356337347527`*^9, 3.825235708731964*^9}, {3.825235791113871*^9, 
   3.8252358110249023`*^9}, {3.825235873050028*^9, 3.8252359240125723`*^9}, {
   3.82523599494002*^9, 3.82523608833853*^9}, {3.8252588702349243`*^9, 
   3.825258944172765*^9}, {3.825259082696916*^9, 3.825259187448353*^9}, {
   3.8252592664019413`*^9, 3.825259600985352*^9}, {3.8252598424053087`*^9, 
   3.825259898091617*^9}, {3.825259931404149*^9, 3.82526022462633*^9}, {
   3.825260268243566*^9, 3.825260289633074*^9}, {3.825260330684801*^9, 
   3.825260364289957*^9}, {3.825260439128838*^9, 3.82526051919656*^9}, {
   3.825260586532983*^9, 3.825260621277328*^9}, {3.825260655749074*^9, 
   3.8252607973710337`*^9}, {3.825260835053217*^9, 3.8252608498085203`*^9}, {
   3.82526091918543*^9, 3.825260938152709*^9}, {3.825260977130481*^9, 
   3.825261031921667*^9}, {3.825265102970092*^9, 3.825265201579459*^9}, {
   3.825265250594282*^9, 3.8252652917285767`*^9}, {3.825265370498365*^9, 
   3.825265401927437*^9}, {3.825265472639279*^9, 3.8252655273575287`*^9}, {
   3.825265694007269*^9, 3.8252657060787086`*^9}, {3.825265783261961*^9, 
   3.82526580614515*^9}, {3.825265854993775*^9, 3.825266062692752*^9}, {
   3.825266116006892*^9, 3.825266121344453*^9}, {3.825266435141489*^9, 
   3.825266528844895*^9}, {3.825266559986397*^9, 3.8252667272695007`*^9}, {
   3.8252667953856077`*^9, 3.82526684081852*^9}, {3.825266966816736*^9, 
   3.825267072180385*^9}, {3.825267122308179*^9, 3.825267122755588*^9}, {
   3.825267199215333*^9, 3.825267232989505*^9}, {3.825267829966318*^9, 
   3.8252678414901123`*^9}, {3.825267997506654*^9, 3.825268008006174*^9}, {
   3.8252680946668386`*^9, 3.825268147373446*^9}, {3.8252682588225403`*^9, 
   3.8252683474593697`*^9}, {3.8252684276904707`*^9, 3.82526852996453*^9}, {
   3.825268961796517*^9, 3.825269048164955*^9}, 3.8252690890558443`*^9, 
   3.825269348371839*^9, {3.825269514196454*^9, 3.825269544653933*^9}, {
   3.8252696071198473`*^9, 3.8252696195506287`*^9}, {3.825269669679532*^9, 
   3.825269731011314*^9}, {3.8252698595048113`*^9, 3.825269899298327*^9}, {
   3.825270020093377*^9, 3.8252701337984*^9}, {3.825270170552009*^9, 
   3.8252702556448107`*^9}, {3.825270344866776*^9, 3.825270369492322*^9}, 
   3.8252704087307053`*^9, {3.8252805145758333`*^9, 3.825280522811632*^9}, {
   3.825280676482954*^9, 3.825280746414288*^9}, {3.82528080300741*^9, 
   3.8252808662553463`*^9}, {3.8252809212407513`*^9, 3.825280924080353*^9}, {
   3.8252810891809196`*^9, 3.825281217438335*^9}, {3.825281251001046*^9, 
   3.825281279010963*^9}, {3.8252813226812077`*^9, 3.8252813444080563`*^9}, {
   3.825281385753591*^9, 3.825281393353766*^9}, {3.825281450017614*^9, 
   3.825281505005143*^9}, {3.8253339323425007`*^9, 3.825333938487838*^9}, {
   3.825334642417944*^9, 3.8253347692256107`*^9}, {3.825334799774823*^9, 
   3.82533483599541*^9}, {3.825334868328405*^9, 3.8253349305892277`*^9}, {
   3.825334964477313*^9, 3.825335031661532*^9}, {3.8253350626309347`*^9, 
   3.8253353492594233`*^9}, {3.825335406846674*^9, 3.825335546187202*^9}, {
   3.8253358145764723`*^9, 3.825335832482811*^9}, {3.8253359729820337`*^9, 
   3.8253360527230883`*^9}, {3.82533612695853*^9, 3.825336127810169*^9}, {
   3.825336615149961*^9, 3.8253367738393497`*^9}, {3.825336835491909*^9, 
   3.8253368381649323`*^9}, {3.825337208664792*^9, 3.825337211342051*^9}, {
   3.82533726604375*^9, 3.825337269097746*^9}, 3.8253373576019297`*^9, {
   3.8253455898061333`*^9, 3.825345609639378*^9}, {3.825347844578125*^9, 
   3.82534824868043*^9}, {3.8253491999177637`*^9, 3.825350093044662*^9}, {
   3.82535012686563*^9, 3.825350240718246*^9}, {3.8253502759749126`*^9, 
   3.825350280116734*^9}, {3.825350520631321*^9, 3.8253505737961607`*^9}, {
   3.8253506536833267`*^9, 3.8253506919248238`*^9}, {3.8253507998502827`*^9, 
   3.825350999582068*^9}, {3.825351033170911*^9, 3.825351053855167*^9}, {
   3.8253510849532127`*^9, 3.8253512233129873`*^9}, {3.825351255230238*^9, 
   3.8253512585786457`*^9}, 3.8253513181841993`*^9, {3.8253513632291307`*^9, 
   3.825351418382839*^9}, {3.8253514716463842`*^9, 3.825351491552783*^9}, {
   3.8253515288293447`*^9, 3.825351539371698*^9}, {3.825351680850716*^9, 
   3.82535169076761*^9}, {3.825351985276134*^9, 3.825352017745899*^9}, {
   3.825352075342649*^9, 3.825352077314999*^9}, {3.825352123059834*^9, 
   3.825352195466567*^9}, {3.8253522481358*^9, 3.825352263856632*^9}, {
   3.825352303623563*^9, 3.8253523359393187`*^9}, {3.82535262929617*^9, 
   3.8253526404933567`*^9}, {3.825352672419025*^9, 3.8253527001586227`*^9}, {
   3.825352730668622*^9, 3.82535277091327*^9}, {3.825352879827939*^9, 
   3.825352880369926*^9}, {3.825353020711104*^9, 3.8253530566241083`*^9}, {
   3.8253530955345287`*^9, 3.825353123000749*^9}, {3.8253532315775633`*^9, 
   3.825353303623988*^9}, {3.8253534348427353`*^9, 3.8253534351968737`*^9}, {
   3.82535351770824*^9, 3.825353527613524*^9}, {3.825353757464097*^9, 
   3.825353827148131*^9}, {3.8253539665933857`*^9, 3.8253539687191687`*^9}, {
   3.82535414916912*^9, 3.825354150953952*^9}, {3.825354197850141*^9, 
   3.8253542082193623`*^9}, 3.825354322432482*^9, {3.825354378233573*^9, 
   3.825354384039259*^9}, {3.825354443670343*^9, 3.825354444591456*^9}, 
   3.825354488241107*^9, {3.8253545241348886`*^9, 3.825354546618279*^9}, {
   3.825363196231257*^9, 3.825363202992743*^9}, {3.825363322870029*^9, 
   3.82536375321218*^9}, {3.827820051746458*^9, 3.827820080681666*^9}, {
   3.8278204892668247`*^9, 3.827820507702236*^9}, {3.8278258775159082`*^9, 
   3.8278259862613707`*^9}, {3.827826210410852*^9, 3.827826234558861*^9}, {
   3.827826280577503*^9, 3.827826499678196*^9}, {3.8278265329188633`*^9, 
   3.827826578686166*^9}, {3.827826719215622*^9, 3.827826730242277*^9}, {
   3.827826833098157*^9, 3.82782686482756*^9}, {3.827833190239966*^9, 
   3.827833482199492*^9}, 3.827833513719171*^9, {3.827836583028665*^9, 
   3.827836588325515*^9}, 3.8278366687932997`*^9, {3.8278367760752497`*^9, 
   3.827836783850375*^9}, 3.827836827456875*^9, {3.8278369002035637`*^9, 
   3.827836930263345*^9}, 3.8278374151716137`*^9, {3.827837669867738*^9, 
   3.827837679178026*^9}, {3.827837760440778*^9, 3.827837967997566*^9}, {
   3.8278380064793053`*^9, 3.827838032690247*^9}, 3.827838078369969*^9, {
   3.827838262127823*^9, 3.8278382920965776`*^9}, 3.827838746265151*^9, {
   3.827839571012435*^9, 3.827839591358585*^9}, {3.827840418679887*^9, 
   3.827840419233869*^9}, {3.827841370760923*^9, 3.827841400910469*^9}, {
   3.827841491914937*^9, 3.827841525224782*^9}, {3.827842840922159*^9, 
   3.827842844036996*^9}, {3.827856977164762*^9, 3.827857104118196*^9}, {
   3.827857221860537*^9, 3.827857256434462*^9}, {3.8278574128993387`*^9, 
   3.827857429875684*^9}, {3.8278579811154013`*^9, 3.827857989521287*^9}, 
   3.827858363821041*^9, {3.828164430832631*^9, 3.8281644376529207`*^9}, {
   3.828164527529539*^9, 3.828164593357976*^9}, 3.82816464314624*^9, {
   3.828164745139785*^9, 3.828164747956944*^9}, {3.828164800847789*^9, 
   3.828164820018993*^9}, 3.828164867054249*^9, 3.828164947440443*^9, {
   3.828165017391203*^9, 3.828165022205241*^9}, {3.8281722773405237`*^9, 
   3.8281722962900343`*^9}, {3.8282049247108927`*^9, 3.828205358697194*^9}, {
   3.828205389641251*^9, 3.828205475830193*^9}, {3.828214143489615*^9, 
   3.828214143623355*^9}, {3.828214233603622*^9, 3.828214263699375*^9}, {
   3.828214354995922*^9, 3.8282144235969267`*^9}, {3.828214478995091*^9, 
   3.8282145095292788`*^9}, {3.828214546445176*^9, 3.82821463197567*^9}, {
   3.8282146832429667`*^9, 3.82821486034568*^9}, {3.828214905067371*^9, 
   3.828214913984621*^9}, {3.8282149542337017`*^9, 3.8282149583712683`*^9}, {
   3.828215017048123*^9, 3.828215263151643*^9}, {3.828215297215803*^9, 
   3.828215351613637*^9}, {3.828215441570836*^9, 3.828215659291143*^9}, {
   3.828215699373837*^9, 3.828215909213416*^9}, {3.828215951812991*^9, 
   3.828216114909203*^9}, 3.828287863152787*^9, {3.828287949467863*^9, 
   3.82828795325119*^9}, {3.8282880036728773`*^9, 3.82828800816385*^9}, {
   3.828288102405253*^9, 3.828288105507635*^9}, {3.8282882439119062`*^9, 
   3.828288272639058*^9}, {3.828288341122656*^9, 3.828288384535008*^9}, {
   3.8282885760687113`*^9, 3.828288610363743*^9}, {3.8282886716533833`*^9, 
   3.828288674988456*^9}, 3.828288731688244*^9, 3.8282888532853937`*^9, {
   3.828288924485486*^9, 3.828288949178999*^9}, {3.828289564542656*^9, 
   3.828289564624321*^9}, {3.828289601332054*^9, 3.828289604502942*^9}, {
   3.828289648707656*^9, 3.828289651360334*^9}, {3.828289707483766*^9, 
   3.828289740548411*^9}, {3.828289977324029*^9, 3.82829009716745*^9}, {
   3.828290141100751*^9, 3.828290361118658*^9}, {3.828290485743256*^9, 
   3.8282904863359547`*^9}, {3.8282905791162643`*^9, 3.828290588112608*^9}, {
   3.828291821055626*^9, 3.828291841118779*^9}, {3.828432278409523*^9, 
   3.8284323569257097`*^9}, {3.8284324686419477`*^9, 3.828432479148774*^9}, {
   3.828432555706801*^9, 3.828432769229991*^9}, {3.828433146460952*^9, 
   3.828433211120893*^9}, {3.8284361561388273`*^9, 3.8284361613861837`*^9}, {
   3.8284364697558823`*^9, 3.828436517877252*^9}, {3.828436567241165*^9, 
   3.828436582716627*^9}, {3.8284367349226837`*^9, 3.8284367413393507`*^9}, 
   3.828436787302948*^9, {3.828443022997188*^9, 3.8284430987284603`*^9}, 
   3.82844312889079*^9, {3.82844364049279*^9, 3.828443728576763*^9}, {
   3.8284438489498663`*^9, 3.828443976073022*^9}, {3.828444008290475*^9, 
   3.8284440281101913`*^9}, {3.8284440994907303`*^9, 3.828444111137124*^9}, {
   3.8284441523247423`*^9, 3.828444178783289*^9}, {3.828444358773636*^9, 
   3.828444381328726*^9}, {3.828444416452619*^9, 3.8284445188886633`*^9}, {
   3.828444623332068*^9, 3.828444626839733*^9}, {3.828444718080071*^9, 
   3.828444782131876*^9}, {3.828444815170994*^9, 3.82844484513822*^9}, {
   3.828444908775158*^9, 3.828444946205386*^9}, {3.828445393395905*^9, 
   3.828445484192786*^9}, {3.828445550979093*^9, 3.828445565972392*^9}, {
   3.8284465687834578`*^9, 3.828446571977109*^9}, {3.828446647557115*^9, 
   3.8284466483832827`*^9}, 3.8284467234234743`*^9, {3.828446777348702*^9, 
   3.8284467886441193`*^9}, {3.828446840759686*^9, 3.82844693377732*^9}, {
   3.828447001514991*^9, 3.8284472327851057`*^9}, {3.828447314517205*^9, 
   3.8284473328691998`*^9}, {3.828447424628538*^9, 3.8284477263915443`*^9}, {
   3.828447757964325*^9, 3.828447761043289*^9}, {3.828447794671946*^9, 
   3.828447901621661*^9}, {3.828447952637031*^9, 3.8284479851643677`*^9}, {
   3.8284480244714813`*^9, 3.828448113118685*^9}, {3.828448144788649*^9, 
   3.828448212951688*^9}, {3.828448249361925*^9, 3.8284487108216476`*^9}, {
   3.828448757857181*^9, 3.8284489012963123`*^9}, {3.8284490301081553`*^9, 
   3.8284491321855164`*^9}, 3.828449198880023*^9, {3.8284492544403276`*^9, 
   3.828449351204026*^9}, {3.828449391260028*^9, 3.828449510974011*^9}, {
   3.82845974978787*^9, 3.828459784726948*^9}, {3.828459964679266*^9, 
   3.828459978469516*^9}, {3.82846004177577*^9, 3.828460052783135*^9}, {
   3.828460150627554*^9, 3.8284602360924053`*^9}, {3.828460627259058*^9, 
   3.828460674732149*^9}, {3.828460776277751*^9, 3.828460859434803*^9}, 
   3.8284608925405893`*^9, {3.828460980033717*^9, 3.828461075383629*^9}, {
   3.8284611371589603`*^9, 3.8284612023061*^9}, {3.8284612408778143`*^9, 
   3.82846124462035*^9}, 3.82846236107511*^9, {3.828462396374094*^9, 
   3.82846241047218*^9}, {3.828462459625966*^9, 3.828462460036338*^9}, {
   3.8284628237028437`*^9, 3.8284628550050488`*^9}, 3.828462897009685*^9, {
   3.828463605381775*^9, 3.828463608532631*^9}, {3.828464034309664*^9, 
   3.8284640750234823`*^9}, {3.828471074625175*^9, 3.828471090155435*^9}, {
   3.8284712039074297`*^9, 3.828471211054532*^9}, {3.8285082526897783`*^9, 
   3.828508274779117*^9}, {3.828520289237014*^9, 3.8285202896693983`*^9}, {
   3.82852036964458*^9, 3.8285203983578*^9}, {3.828520430751585*^9, 
   3.828520708849841*^9}, {3.828520739936964*^9, 3.828520775821224*^9}, {
   3.828521018089016*^9, 3.828521304377469*^9}, {3.828521336961544*^9, 
   3.828521856753927*^9}, {3.82852189040447*^9, 3.8285219200114193`*^9}, {
   3.828521992690222*^9, 3.828522093023814*^9}, {3.828522134133913*^9, 
   3.8285223699113083`*^9}, {3.82852243802702*^9, 3.8285224411245728`*^9}, 
   3.828522486437545*^9, {3.8285225316583433`*^9, 3.828522555231412*^9}, {
   3.828522597131871*^9, 3.828522867709306*^9}, {3.828522913578044*^9, 
   3.828523173973783*^9}, {3.828523218380427*^9, 3.828523299970326*^9}, {
   3.828523332517817*^9, 3.828523334936276*^9}, {3.828523394928176*^9, 
   3.828523486433975*^9}, {3.828523704591606*^9, 3.82852420150078*^9}, {
   3.828524248150345*^9, 3.828524302884934*^9}, {3.828524394616745*^9, 
   3.8285244210673637`*^9}, {3.828528601339971*^9, 3.8285287608063793`*^9}, {
   3.828528845588153*^9, 3.82852895674465*^9}, {3.82852900255641*^9, 
   3.8285290900660152`*^9}, {3.828529126885334*^9, 3.828529170226191*^9}, {
   3.828529204418743*^9, 3.828529220618585*^9}, {3.8285292587602453`*^9, 
   3.828529269144554*^9}, {3.828529314051166*^9, 3.828529466405349*^9}, {
   3.828529546860375*^9, 3.8285295920202703`*^9}, {3.8285296282599077`*^9, 
   3.8285298643353167`*^9}, {3.828529915518195*^9, 3.828529935135942*^9}, {
   3.828530017749645*^9, 3.828530148686389*^9}, {3.828530200430293*^9, 
   3.828530492584578*^9}, {3.828530525913484*^9, 3.828530560461179*^9}, {
   3.828530596186469*^9, 3.828530608125435*^9}, {3.828530647101369*^9, 
   3.828530808530777*^9}, {3.828531041205225*^9, 3.8285310912878437`*^9}, {
   3.8329936857441177`*^9, 3.832993705578217*^9}, {3.832993894518605*^9, 
   3.832994045183177*^9}, {3.8329941005912123`*^9, 3.8329941148278017`*^9}, {
   3.8329941789160643`*^9, 3.832994185496973*^9}, {3.83299527393016*^9, 
   3.832995381399555*^9}, {3.832995414138756*^9, 3.832995448921542*^9}, {
   3.832995509792617*^9, 3.832995548576858*^9}, 3.8329961250791473`*^9, {
   3.833006394336514*^9, 3.833006562176897*^9}, {3.8330068350466967`*^9, 
   3.8330068856475697`*^9}, {3.833006956540625*^9, 3.833007141214896*^9}, 
   3.833007198792144*^9, {3.833007774152452*^9, 3.833007802740735*^9}, {
   3.833007842015522*^9, 3.833007869332872*^9}, {3.833007957576989*^9, 
   3.833007960087102*^9}, {3.833009743153832*^9, 3.83300978177145*^9}, {
   3.833009893792275*^9, 3.8330099010117693`*^9}, {3.8330101705369368`*^9, 
   3.833010269853595*^9}, {3.8330103258663816`*^9, 3.833010329305336*^9}, {
   3.833010384684991*^9, 3.833010419524122*^9}, {3.833010678727044*^9, 
   3.833010688174244*^9}, {3.833012253403948*^9, 3.833012495741946*^9}, {
   3.833012663292419*^9, 3.833012665368573*^9}, {3.833012811699074*^9, 
   3.833012812185635*^9}, {3.833012848811907*^9, 3.833012849281578*^9}, {
   3.833013070680915*^9, 3.833013070943746*^9}, {3.8330131026705637`*^9, 
   3.833013114359852*^9}, {3.8332808977449903`*^9, 3.8332808979902897`*^9}, {
   3.8332841817044897`*^9, 3.833284203047406*^9}, {3.833284355202722*^9, 
   3.833284377738614*^9}, {3.833284658059718*^9, 3.833284661860359*^9}, {
   3.833284860706455*^9, 3.8332849245565023`*^9}, {3.833285054875464*^9, 
   3.833285081578541*^9}, {3.8332915480723457`*^9, 3.833291565240456*^9}, {
   3.8333612443615*^9, 3.833361532435874*^9}, {3.83336156813061*^9, 
   3.8333615773728456`*^9}, {3.833361707703858*^9, 3.833361757721242*^9}, {
   3.833361846750683*^9, 3.833361868362885*^9}, {3.833362120613324*^9, 
   3.833362121210743*^9}, {3.8333621558795433`*^9, 3.833362174183565*^9}, {
   3.833362225615933*^9, 3.8333622358599854`*^9}, {3.833362333348136*^9, 
   3.833362362075932*^9}, {3.83336245095644*^9, 3.833362458656837*^9}, {
   3.833362517558382*^9, 3.833362517849106*^9}, {3.833363516296665*^9, 
   3.8333635537961273`*^9}, {3.8333639902141333`*^9, 
   3.8333639952762203`*^9}, {3.833364357688245*^9, 3.833364421224125*^9}, {
   3.833602116895067*^9, 3.833602130741218*^9}, {3.833610034827033*^9, 
   3.8336100360305767`*^9}, 3.833611181548962*^9, {3.833957661273057*^9, 
   3.833957674813025*^9}, 3.833967810870741*^9, {3.8339859470893383`*^9, 
   3.833985951720015*^9}, {3.833989378092073*^9, 3.833989491514206*^9}, 
   3.834030286125185*^9, {3.834030466297757*^9, 3.834030476938689*^9}, {
   3.8340320269659567`*^9, 3.834032071568797*^9}, {3.834032170580318*^9, 
   3.834032207468157*^9}, {3.83403228270439*^9, 3.834032301934011*^9}, {
   3.834032410763644*^9, 3.834032446316038*^9}, {3.834032717083015*^9, 
   3.834032720563517*^9}, {3.8340331841987333`*^9, 3.834033203630925*^9}, {
   3.834033316423606*^9, 3.834033320014265*^9}, {3.8340334225103407`*^9, 
   3.834033423639257*^9}, {3.834033679262788*^9, 3.8340337465054007`*^9}, {
   3.834033789123394*^9, 3.8340338056090393`*^9}, {3.834033912389434*^9, 
   3.834033932804678*^9}, {3.834034067200964*^9, 3.8340340988188267`*^9}, {
   3.834037082438051*^9, 3.834037097944221*^9}, {3.8340371556299562`*^9, 
   3.8340371573818483`*^9}, {3.834042549220854*^9, 3.8340426002391243`*^9}, {
   3.834042632899437*^9, 3.8340426528385344`*^9}, {3.8340428083617067`*^9, 
   3.834042877993414*^9}, {3.83404290911722*^9, 3.834042957101233*^9}, {
   3.8340430086584883`*^9, 3.8340432136258917`*^9}, {3.834043419237679*^9, 
   3.834043455721953*^9}, 3.834043489895171*^9, {3.834043667805635*^9, 
   3.8340436771524878`*^9}, {3.834043741446286*^9, 3.834043764552696*^9}, {
   3.8340439120982523`*^9, 3.834043925487862*^9}, {3.834044113585375*^9, 
   3.83404413305961*^9}, {3.83404433899657*^9, 3.83404434265489*^9}, {
   3.8340445018271217`*^9, 3.834044512386105*^9}, {3.8340445818124332`*^9, 
   3.834044617791193*^9}, {3.8340451643029547`*^9, 3.834045180262743*^9}, {
   3.834045918305421*^9, 3.834045933361148*^9}, {3.834046001126031*^9, 
   3.834046009689044*^9}, {3.834046054526177*^9, 3.834046073474162*^9}, {
   3.834046120341201*^9, 3.834046130714076*^9}, {3.834046362559163*^9, 
   3.834046367794319*^9}, {3.8340466731389713`*^9, 3.834046687970953*^9}, {
   3.834046777572154*^9, 3.834046778459887*^9}, {3.834046901131908*^9, 
   3.834046906980443*^9}, {3.834047540226984*^9, 3.83404756196721*^9}, {
   3.834050148117558*^9, 3.834050155329369*^9}, {3.834050361244254*^9, 
   3.834050361512712*^9}, 3.834050435469858*^9, {3.834051296344953*^9, 
   3.834051335497889*^9}, 3.834057982343092*^9, {3.834058096738469*^9, 
   3.83405812062574*^9}, {3.834058340069859*^9, 3.83405836961497*^9}, {
   3.834058403618758*^9, 3.834058404883911*^9}, {3.834058528159688*^9, 
   3.834058538483614*^9}, {3.835609892447912*^9, 3.835609920201408*^9}, {
   3.835610048227178*^9, 3.835610052244849*^9}, {3.835610139958229*^9, 
   3.835610178710394*^9}, {3.835610276650057*^9, 3.835610313160591*^9}, {
   3.835610355249395*^9, 3.8356103673771057`*^9}, {3.835681339628016*^9, 
   3.8356813672511063`*^9}, {3.83568163360994*^9, 3.835681759028452*^9}, {
   3.835681793037826*^9, 3.8356817931570997`*^9}, {3.835681826658909*^9, 
   3.835681874079178*^9}, {3.8356820026506147`*^9, 3.835682044208692*^9}, {
   3.835682180579995*^9, 3.8356821825923243`*^9}, {3.835682225756495*^9, 
   3.835682247490979*^9}, {3.835682533333189*^9, 3.8356825340126953`*^9}, {
   3.835682709883135*^9, 3.835682729497692*^9}, {3.8356828658575573`*^9, 
   3.835682911100237*^9}, {3.835682988444572*^9, 3.8356830071983547`*^9}, 
   3.835683607350518*^9, {3.8356838542672377`*^9, 3.835683856910408*^9}, {
   3.835683995585746*^9, 3.83568401674681*^9}, {3.835684203108302*^9, 
   3.835684214581456*^9}, {3.8356881012168093`*^9, 3.835688131040103*^9}, 
   3.83568824818723*^9, {3.83568830204601*^9, 3.835688412217198*^9}, {
   3.835688484466896*^9, 3.835688487389743*^9}, {3.835695679681693*^9, 
   3.835695709080621*^9}, {3.835695770722797*^9, 3.8356957747780657`*^9}, {
   3.835695814799409*^9, 3.835695822496068*^9}, {3.835695880339178*^9, 
   3.8356959449825373`*^9}, {3.835696858039933*^9, 3.8356968823379917`*^9}, {
   3.8357009380386667`*^9, 3.835700959164589*^9}, {3.8357010084484587`*^9, 
   3.83570104226066*^9}, {3.835936215896337*^9, 3.835936289578326*^9}, {
   3.835936343390818*^9, 3.8359365560321302`*^9}, {3.83594440900214*^9, 
   3.8359445699831553`*^9}, {3.835945291344602*^9, 3.835945291503787*^9}, {
   3.8366246615826683`*^9, 3.836624793109271*^9}, {3.8366249328315287`*^9, 
   3.836625007585226*^9}, {3.8366257106276283`*^9, 3.836625715521618*^9}, {
   3.8366257548040123`*^9, 3.836625827729309*^9}, {3.836625858152905*^9, 
   3.836625864045689*^9}, {3.836625900349465*^9, 3.83662590350826*^9}, {
   3.836628699146371*^9, 3.836628784968816*^9}, {3.83662994618557*^9, 
   3.836630022960124*^9}, {3.836630737151629*^9, 3.836630746457616*^9}, {
   3.836630903448107*^9, 3.836631011836315*^9}, {3.836631994646174*^9, 
   3.836632001425323*^9}, {3.836632105816733*^9, 3.8366321391510057`*^9}, {
   3.836633699459906*^9, 3.836633700311714*^9}, {3.836634557547168*^9, 
   3.836634619707863*^9}, {3.8366346611616917`*^9, 3.836634663660101*^9}, {
   3.836634705914363*^9, 3.8366347735874233`*^9}, {3.8366351961142178`*^9, 
   3.8366352270892982`*^9}, {3.836635299952034*^9, 3.83663530532481*^9}, {
   3.836635349230585*^9, 3.836635352558889*^9}, {3.836635393409595*^9, 
   3.8366354013939543`*^9}, {3.836635643613648*^9, 3.8366356783948517`*^9}, 
   3.836635718888912*^9, {3.83663968653279*^9, 3.8366396875244513`*^9}, {
   3.836643379593092*^9, 3.836643498202001*^9}, {3.836643557867099*^9, 
   3.8366437635726357`*^9}, {3.836643799059833*^9, 3.8366438000106583`*^9}, {
   3.836643890391303*^9, 3.836644192003302*^9}, {3.836669718149572*^9, 
   3.836669806250918*^9}, {3.836670337869459*^9, 3.836670539184518*^9}, {
   3.836899355951412*^9, 3.836899356219254*^9}, {3.836899619079262*^9, 
   3.836899636319268*^9}, {3.836900084347064*^9, 3.836900088600884*^9}, {
   3.8369001811986303`*^9, 3.8369001993722887`*^9}, {3.836900331743588*^9, 
   3.836900362848502*^9}, {3.836900422263028*^9, 3.836900443402317*^9}, {
   3.836900497269022*^9, 3.8369005119325943`*^9}, 3.83690065015845*^9, 
   3.836900698839102*^9, {3.836901051559513*^9, 3.8369010577943487`*^9}, 
   3.836901787733327*^9, {3.83696537823197*^9, 3.8369653855818167`*^9}, {
   3.836965780641711*^9, 3.8369657879489613`*^9}, {3.836965850127633*^9, 
   3.83696589483185*^9}, {3.8369659891611633`*^9, 3.8369659923141747`*^9}, {
   3.836966031766856*^9, 3.836966090695054*^9}, {3.836967072857594*^9, 
   3.836967073050828*^9}, {3.836967451254025*^9, 3.83696753106441*^9}, {
   3.836968143472104*^9, 3.8369681931399317`*^9}, {3.8369682749808683`*^9, 
   3.83696830424461*^9}, {3.836968494904687*^9, 3.8369685860894623`*^9}, 
   3.836968627850999*^9, {3.836968677700109*^9, 3.8369687208628607`*^9}, {
   3.836968851335371*^9, 3.8369688579787893`*^9}, {3.836969034584736*^9, 
   3.8369690374497538`*^9}, {3.8369692252787933`*^9, 
   3.8369692313657637`*^9}, {3.836969621051243*^9, 3.836969641413844*^9}, {
   3.836969864547175*^9, 3.836969869685557*^9}, {3.836969918654975*^9, 
   3.836969921399527*^9}, {3.8369703852573977`*^9, 3.8369704142883253`*^9}, {
   3.8374338607228947`*^9, 3.837433864255556*^9}, {3.837496801854704*^9, 
   3.8374968143359222`*^9}, {3.837496916887041*^9, 3.83749693697295*^9}, {
   3.83757213641029*^9, 3.837572150813253*^9}, {3.83757235473125*^9, 
   3.837572367996814*^9}, 3.8414080712712393`*^9, 3.841408148732416*^9, {
   3.841482724061968*^9, 3.841482745145074*^9}, {3.841482776478948*^9, 
   3.8414828252296553`*^9}, {3.8414836781692247`*^9, 3.841483679971611*^9}, {
   3.841488012448674*^9, 3.841488016238076*^9}, {3.8478092396153708`*^9, 
   3.847809276621814*^9}, {3.8478102982832813`*^9, 3.847810308558158*^9}, {
   3.847810342551093*^9, 3.8478104462553062`*^9}, {3.847810476768764*^9, 
   3.847810510909605*^9}, {3.847811124177286*^9, 3.847811134942083*^9}, {
   3.8478130607263412`*^9, 
   3.847813094614151*^9}},ExpressionUUID->"b6039133-1016-473e-b566-\
8b2e266ea331"],

Cell[BoxData[
 RowBox[{
  RowBox[{"EndPackage", "[", "]"}], ";"}]], "Input",
 InitializationCell->True,
 CellChangeTimes->{{3.84781295393104*^9, 
  3.8478129569389544`*^9}},ExpressionUUID->"e038966a-26b6-495e-9a70-\
a20019c0f147"]
},
WindowSize->{1440, 801},
WindowMargins->{{Automatic, -138}, {-45, Automatic}},
TaggingRules->Association["TryRealOnly" -> False],
FrontEndVersion->"12.3 for Mac OS X ARM (64-bit) (July 9, 2021)",
StyleDefinitions->"Default.nb",
ExpressionUUID->"4dbf9779-7c48-4d22-beee-d282c8fb97df"
]
(* End of Notebook Content *)

(* Internal cache information *)
(*CellTagsOutline
CellTagsIndex->{}
*)
(*CellTagsIndex
CellTagsIndex->{}
*)
(*NotebookFileOutline
Notebook[{
Cell[558, 20, 288, 6, 46, "Input",ExpressionUUID->"b7c0a472-a1ce-4fc3-84c0-7ff1a299e1b7",
 InitializationCell->True],
Cell[849, 28, 796, 18, 87, "Input",ExpressionUUID->"0d1c205d-9948-4aa4-a8e4-3bd5ead1ac02",
 InitializationCell->True],
Cell[1648, 48, 154073, 3215, 13251, "Input",ExpressionUUID->"b6039133-1016-473e-b566-8b2e266ea331",
 InitializationCell->True],
Cell[155724, 3265, 230, 6, 46, "Input",ExpressionUUID->"e038966a-26b6-495e-9a70-a20019c0f147",
 InitializationCell->True]
}
]
*)

