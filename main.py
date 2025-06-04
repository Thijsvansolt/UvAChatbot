import asyncio
import streamlit as st
from ui.stream_ui import main as stream_ui_main

if __name__ == "__main__":
    asyncio.run(stream_ui_main())
