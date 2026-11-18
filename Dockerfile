# The page's container: the React client built in one stage, the FastAPI service in the next.
# The forecast jobs (GFS fetch, pair logging, inference) do not run here. They run in
# GitHub Actions from requirements.txt and commit what this container reads.

FROM node:22-slim AS client
WORKDIR /client
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
# Only what the service imports: no grib stack, so the image needs no system libraries.
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt
COPY src/ src/
COPY service/ service/
COPY data/conus_heat_tracker.nc data/conus_heat_tracker.nc
COPY --from=client /client/dist client/dist

# Render finds the port a Docker service listens on; 8051 is the one this service has always used.
EXPOSE 8051
# One process: the forecast is held in memory once, and the routes run in its thread pool.
# Keep-alive: Render's proxy keeps connections to the service open and reuses them. uvicorn
# closes an idle one after 5 seconds by default, and a request sent down a connection it has
# just closed comes back to the reader as a 502. Seen on the first day live: about four
# requests in ten. The service now holds idle connections longer than the proxy does.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8051", "--timeout-keep-alive", "120"]
