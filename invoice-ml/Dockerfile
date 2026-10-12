
FROM node:22-alpine

WORKDIR /app

COPY train.js ./train.js

ENV NODE_ENV=production
ENV TRAINING_DATA_PATH=/app/data/invoices.csv
ENV MODEL_OUTPUT_DIR=/app/output

RUN mkdir -p /app/data /app/output

USER node

CMD ["node", "train.js"]
