call "%HOMEPATH%\anaconda3\Scripts\activate.bat" diamond_needle & cd %~dp0 & pyinstaller --add-data="resourse;resourse" --onefile --noconsole --add-data="font;font" src/main.py
pause>nul