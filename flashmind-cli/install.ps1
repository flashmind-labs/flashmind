& {
    $ProgressPreference = 'SilentlyContinue'
    $ErrorActionPreference = 'Stop'
    $base = 'https://releases.flashmind.com/flashmind/cli'
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'No flsh binary is available for this platform.' }
    $version = (Invoke-RestMethod "$base/latest.txt").Trim()
    if ($version -notmatch '^\d+\.\d+\.\d+$') { throw 'Invalid release version.' }
    $name = 'flsh-x86_64-pc-windows-msvc.exe'
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
    try {
        Invoke-WebRequest "$base/$version/$name" -OutFile $tmp
        $expected = (Invoke-RestMethod "$base/$version/$name.sha256").Trim()
        if ((Get-FileHash -Algorithm SHA256 $tmp).Hash -ne $expected) { throw 'Checksum mismatch.' }
        $dir = if ($env:FLSH_INSTALL_DIR) { $env:FLSH_INSTALL_DIR } else { Join-Path $env:USERPROFILE '.local\bin' }
        New-Item -ItemType Directory -Force $dir | Out-Null
        Move-Item -Force $tmp (Join-Path $dir 'flsh.exe')
        Write-Host "Installed flsh $version to $dir\flsh.exe"
        Write-Host "Add $dir to PATH if it is not already there."
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $tmp
    }
}
