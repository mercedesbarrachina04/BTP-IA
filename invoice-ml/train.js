
'use strict';

const fs = require('fs');
const path = require('path');

const categories = [
  'Software', 'Services', 'Materials', 'Logistics'
];

// Same preprocessing as the Cloud Foundry application
function tokenize(text) {
  return (String(text || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .match(/[a-z0-9]+/g) || [])
    .filter(word => word.length > 1);
}

// Multinomial Naive Bayes training
function fit(rows) {
  if (!Array.isArray(rows) || rows.length < 8) {
    throw Error('At least 8 labeled invoices required.');
  }

  const byClass = Object.fromEntries(
    categories.map(c => [
      c, { n: 0, total: 0, counts: {} }
    ])
  );

  const vocabulary = new Set();

  for (const invoice of rows) {
    if (!invoice.description ||
        !categories.includes(invoice.category)) {
      throw Error('Invalid invoice record.');
    }

    const c = byClass[invoice.category];
    c.n++;

    for (const word of tokenize(invoice.description)) {
      c.counts[word] = (c.counts[word] || 0) + 1;
      c.total++;
      vocabulary.add(word);
    }
  }

  for (const label of categories) {
    if (byClass[label].n < 2) {
      throw Error('At least 2 records per category required.');
    }
  }

  return {
    byClass,
    vocabSize: vocabulary.size,
    n: rows.length
  };
}

// CSV reader supporting quoted fields
function csvRows(input) {
  const rows = [];
  let row = [], cell = '', quoted = false;
  const s = String(input).replace(/^\uFEFF/, '');

  for (let i = 0; i < s.length; i++) {
    const ch = s[i];

    if (quoted) {
      if (ch === '"' && s[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      row.push(cell);
      cell = '';
    } else if (ch === '\n') {
      row.push(cell.replace(/\r$/, ''));
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
    }
  }

  if (quoted) throw Error('Malformed CSV.');

  if (cell.length || row.length) {
    row.push(cell.replace(/\r$/, ''));
    rows.push(row);
  }

  const header = rows.shift()?.map(x =>
    x.trim().toLowerCase()
  ) || [];

  const di = header.indexOf('description');
  const ci = header.indexOf('category');

  if (di < 0 || ci < 0) {
    throw Error('CSV needs description,category headers.');
  }

  return rows
    .filter(r => r.some(x => x.trim()))
    .map(r => ({
      description: (r[di] || '').trim(),
      category: (r[ci] || '').trim()
    }));
}

// AI Core will provide mounted input/output paths
const inputPath = process.env.TRAINING_DATA_PATH ||
  path.join(__dirname, 'data', 'invoices.csv');

const outputDir = process.env.MODEL_OUTPUT_DIR ||
  path.join(__dirname, 'output');

function main() {
  console.log('Starting Naive Bayes training...');
  console.log('Dataset:', inputPath);

  const csv = fs.readFileSync(inputPath, 'utf8');
  const rows = csvRows(csv);

  const model = fit(rows);

  fs.mkdirSync(outputDir, { recursive: true });

  const outputPath = path.join(
    outputDir, 'invoice-model.json'
  );

  fs.writeFileSync(
    outputPath,
    JSON.stringify(model, null, 2)
  );

  console.log('Training completed.');
  console.log('Invoices:', model.n);
  console.log('Vocabulary:', model.vocabSize);
  console.log('Model artifact:', outputPath);
}

main();
