custom_style = '''
    <style>
        .block-container > div > 
        [data-testid="stVerticalBlock"] > 
        [data-testid="stHorizontalBlock"] > 
        [data-testid="column"]:nth-of-type(1) {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: calc(1em - 1px);
        }
        .block-container > div > 
        [data-testid="stVerticalBlock"] > 
        [data-testid="stHorizontalBlock"] > 
        [data-testid="column"]:nth-of-type(1) > div > 
        [data-testid="stVerticalBlock"] > .element-container:nth-of-type(1) {
            position: absolute;
        }
        .block-container > div > 
        [data-testid="stVerticalBlock"] > 
        [data-testid="stHorizontalBlock"] > 
        [data-testid="column"]:nth-of-type(2) {
            height: 120vh;
            overflow-x: hidden;
            overflow-y: scroll;
        }
        .block-container > div > 
        [data-testid="stVerticalBlock"] > 
        [data-testid="stHorizontalBlock"] > 
        [data-testid="column"]:nth-of-type(2) 
        div:not([data-testid="stExpander"]):not([data-testid="stImage"]) {
            width: 100%!important;
        }
        button:disabled, button:disabled:hover, button:disabled:active {
            border-color: transparent!important;
            color: unset!important;
            cursor: auto!important;
            padding-left: 0px;
        }
        thead {
            display: none;
        }
    </style>
'''