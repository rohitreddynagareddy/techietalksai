FROM continuumio/miniconda3
      # or “mambaorg/micromamba” for a leaner image

RUN conda install -y python=3.11 nodejs \
 && conda clean -afy

WORKDIR /app
COPY . .

# CMD ["bash"]
# 4. Upgrade pip and install Python dependencies (including mcp-use, Playwright)
COPY requirements.txt .  
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt  
 #:contentReference[oaicite:3]{index=3}

# # 5. Install Playwright Python and Node binaries, with dependencies
# RUN pip install playwright npm \
#     # Install OS-level deps for Chromium in CI (e.g. libnss3, fonts, etc.)
#     && npx playwright install-deps chromium \
#     # Download the Chromium browser itself
#     && npx playwright install chromium  
# #:contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

# install OS deps for Chromium, then download Chromium itself
RUN npx playwright install-deps chromium \
 && npx playwright install chromium

RUN mkdir -p /opt/google/chrome \
 && ln -sf /usr/bin/chromium /opt/google/chrome/chrome

# 6. Install the Playwright MCP server globally
# RUN npm install -g @playwright/mcp@latest

#:contentReference[oaicite:6]{index=6}

# 7. Symlink the Debian chromium binary into Playwright’s expected path
# RUN mkdir -p /opt/google/chrome \
#     && ln -s /usr/bin/chromium /opt/google/chrome/chrome  
#:contentReference[oaicite:7]{index=7}

# 8. Copy application code
WORKDIR /app
COPY . .

# 9. Start MCP server in background, then run your Python main
# CMD sh -c "\
#     npx @playwright/mcp@latest & \
#     sleep 1 && \
#     python main.py\
# "

# 6) Default entrypoint: launch the MCP server (Node) in the background,
#    wait a moment for it to spin up, then run your Python script.
# CMD sh -c "\
#     npx @playwright/mcp@latest & \
#     sleep 2 && \
#     python main.py\
# "


# # Start from Node 23 so we have Node pre‑installed
# FROM node:23

# # Install Python 3.11 and venv support
# RUN apt-get update && apt-get install -y \
#     python3.11 python3.11-venv python3-pip \
#   && rm -rf /var/lib/apt/lists/*

# # Make 'python' point to Python 3.11
# RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# # Create and activate a virtual environment
# RUN python -m venv /venv
# ENV PATH="/venv/bin:$PATH"

# # Upgrade pip and install your Python deps (including playwright)
# COPY requirements.txt .
# RUN pip install --upgrade pip setuptools wheel \
#  && pip install --no-cache-dir -r requirements.txt

# # Install Playwright browsers for Python
# RUN pip install playwright npm \
#  && playwright install 

# RUN npm install
# RUN npx playwright install chrome

# # Copy your application code
# WORKDIR /app
# COPY . .

# # By default, launch the MCP server in background, then run your script
# # CMD sh -c "\
# #     npx @playwright/mcp@latest & \
# #     sleep 1 && \
# #     python main.py\
# # "
