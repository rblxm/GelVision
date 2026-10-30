#!/bin/bash
cd "$(dirname "$0")"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run the app and open the browser
streamlit run gelvision/app.py
