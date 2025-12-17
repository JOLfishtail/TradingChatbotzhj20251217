# check_env.ps1 - 检查learnacp环境
Write-Host "Checking learnacp environment..." -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan

# 1. 检查Python版本
Write-Host "`nPython Version:" -ForegroundColor Yellow
python --version

# 2. 检查关键包
Write-Host "`nChecking key packages..." -ForegroundColor Yellow

# 定义要检查的包
$packages = @(
    "langchain",
    "langchain-core",
    "langgraph",
    "openai",
    "chromadb",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pandas",
    "numpy",
    "python-dotenv",
    "aiohttp",
    "sqlalchemy"
)

# 检查每个包
foreach ($pkg in $packages) {
    $output = & pip show $pkg 2>&1
    if ($LASTEXITCODE -eq 0) {
        $version = ($output | Select-String "Version:").ToString().Split(":")[1].Trim()
        Write-Host "  FOUND: $($pkg.PadRight(20)) $version" -ForegroundColor Green
    } else {
        Write-Host "  MISSING: $($pkg.PadRight(20)) Not installed" -ForegroundColor Red
    }
}

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "Environment check complete!" -ForegroundColor Green