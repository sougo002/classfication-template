if "%~x1"==".xlsx" (
  copy %1 %1.zip
  start "" "%~f1.zip\xl\media"
)