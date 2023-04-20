const express = require('express');
const React = require('react');
const ReactDOMServer = require('react-dom/server');

const HelloWorld = require('./HelloWorld');

const app = express();

app.get('/', (req, res) => {
  const html = ReactDOMServer.renderToString(<HelloWorld />);
  res.send(`<!DOCTYPE html>
<html>
  <head>
    <title>Exemple SSR React</title>
  </head>
  <body>
    <div id="app">${html}</div>
  </body>
</html>`);
});

const port = 3000;
app.listen(port, () => {
  console.log(`Serveur fonctionnant sur http://localhost:${port}`);
});
