# Build Stage
FROM node:20-slim as build
WORKDIR /app
COPY package*.json ./
# Switch to yarn for better stability
RUN yarn install --network-timeout 100000
COPY . .
RUN yarn build

# Serve Stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# React Router support (redirect to index.html)
RUN echo 'server { \
    listen 80; \
    location / { \
    root /usr/share/nginx/html; \
    index index.html index.htm; \
    try_files $uri $uri/ /index.html; \
    } \
    }' > /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
