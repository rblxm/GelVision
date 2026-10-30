import streamlit.web.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "gelvision/app.py", "--global.developmentMode=false"]
    sys.exit(stcli.main())
