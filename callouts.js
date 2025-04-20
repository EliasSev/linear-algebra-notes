fetch('test-note.md')
  .then(res => res.text())
  .then(md => {
    const mdIt = window.markdownit({ html: true })
      .use(texmath, { engine: katex, delimiters: 'dollars' })
      .use(md => {
        const defaultRender = md.renderer.rules.blockquote_open || function (tokens, idx, options, env, self) {
          return self.renderToken(tokens, idx, options);
        };

        md.renderer.rules.blockquote_open = function (tokens, idx, options, env, self) {
          const nextToken = tokens[idx + 1];
          if (nextToken && nextToken.type === 'inline' && nextToken.content.startsWith('[!')) {
            const type = nextToken.content.match(/^\[\!(\w+)\]/i)?.[1]?.toLowerCase();
            if (type) {
              tokens[idx].attrSet('class', `callout ${type}`);
              nextToken.content = nextToken.content.replace(/^\[\!(\w+)\]\s*/, '');
            }
          }
          return defaultRender(tokens, idx, options, env, self);
        };
      });

    document.getElementById('content').innerHTML = mdIt.render(md);
  });
