FROM node:20-slim

WORKDIR /app

# Copy package files first to optimize build caching
COPY package*.json .
RUN npm install

# Copy everything else
COPY . .

EXPOSE 5173

# Assumes you are using Vite (adjust if using Create React App)
CMD ["npm", "run", "dev", "--", "--host"]