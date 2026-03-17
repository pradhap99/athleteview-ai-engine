FROM node:20-alpine AS base
WORKDIR /app
COPY package*.json ./
RUN npm ci --production

FROM base AS dev
RUN npm ci
COPY . .
CMD ["npm", "run", "dev"]

FROM base AS build
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS production
WORKDIR /app
COPY --from=base /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json .
EXPOSE 3000
HEALTHCHECK --interval=10s CMD wget -qO- http://localhost:3000/health || exit 1
CMD ["node", "dist/index.js"]
