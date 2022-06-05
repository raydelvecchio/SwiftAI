import streamlit as st
from common.constants import DATA_LOCATION
from common.swiftai import SwiftAI
from datetime import datetime


def get_cache_time():
    """
    Returns day/month/year for use in streamlit cached functions. In order for the function to cache, we must
    give it a unique input to cache for. In this case, we'll be caching every day if we consistently input this.
    """
    return datetime.now().strftime("%d/%m/%Y %H")


@st.cache(allow_output_mutation=True)
def get_SwiftAI(date):
    swift = SwiftAI(DATA_LOCATION)
    return swift
