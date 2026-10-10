; Inno Setup script — compiled in CI:  iscc /DAppVersion=0.1.0 build\installer.iss
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6A3B2F51-7C2D-4E8B-9A10-3C1A55E0D2F7}
AppName=课堂同传
AppVersion={#AppVersion}
AppPublisher=课堂同传
DefaultDirName={localappdata}\Programs\ClassInterpreter
DefaultGroupName=课堂同传
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\release
OutputBaseFilename=ClassInterpreter-Setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\ClassInterpreter.exe
Compression=lzma2/fast
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Files]
Source: "..\dist\ClassInterpreter\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\dist_assets\bin\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{autoprograms}\课堂同传"; Filename: "{app}\ClassInterpreter.exe"
Name: "{autodesktop}\课堂同传"; Filename: "{app}\ClassInterpreter.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式:"

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "正在安装界面组件 (WebView2)…"; Check: NeedsWebView2
Filename: "{app}\ClassInterpreter.exe"; Description: "立即打开课堂同传"; Flags: nowait postinstall skipifsilent

[Code]
function HasWebView2Key(Root: Integer; Key: String): Boolean;
var v: String;
begin
  Result := RegQueryStringValue(Root, Key, 'pv', v) and (v <> '') and (v <> '0.0.0.0');
end;

function NeedsWebView2: Boolean;
var k: String;
begin
  k := '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  Result := not (HasWebView2Key(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + k)
              or HasWebView2Key(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + k)
              or HasWebView2Key(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\' + k));
end;
