/* Table of contents for the target page.
 *
 * Built from the rendered page rather than from the survey registry: which
 * sections a target actually shows depends on its coordinates, on what has
 * been acquired already and on whether a chain is running, so a list built
 * from anything else would drift and end up pointing at sections that are
 * not there.
 */

document.addEventListener('DOMContentLoaded', function() {
  var headings = Array.prototype.slice
      .call(document.querySelectorAll('[data-toc]'))
      .filter(function(heading) { return heading.id; });

  // Not worth its own navigation for a couple of sections
  if (headings.length < 3)
    return;

  var nav = document.createElement('nav');
  nav.id = 'toc';
  nav.className = 'toc';

  var title = document.createElement('div');
  title.className = 'toc-title';
  title.textContent = 'Contents';
  nav.appendChild(title);

  var list = document.createElement('ul');
  list.className = 'nav flex-column';

  headings.forEach(function(heading) {
    var item = document.createElement('li');
    item.className = 'nav-item';

    var link = document.createElement('a');
    link.className = 'nav-link';
    link.href = '#' + heading.id;
    // The short name, with the full one left for the tooltip - the rail is
    // deliberately too narrow for 'Mini-MegaTORTORA'
    link.textContent = heading.dataset.toc || heading.textContent.trim();
    // The heading holds a status badge as well, so the full name is carried
    // separately rather than scraped back out of it
    link.title = heading.dataset.tocTitle || heading.textContent.trim();

    item.appendChild(link);
    list.appendChild(item);
  });

  nav.appendChild(list);
  document.body.appendChild(nav);

  // Where the window is too narrow for the rail to sit beside the content,
  // the same list is reached through a button instead
  var toggle = document.createElement('button');
  toggle.className = 'toc-toggle btn btn-sm btn-outline-secondary';
  toggle.type = 'button';
  toggle.title = 'Contents';
  toggle.innerHTML = '<i class="fa fa-list"></i>';
  document.body.appendChild(toggle);

  toggle.addEventListener('click', function() {
    nav.classList.toggle('toc-open');
  });

  // Opened over the page, it should get out of the way once it has been used
  nav.addEventListener('click', function(event) {
    if (event.target.closest('.nav-link'))
      nav.classList.remove('toc-open');
  });

  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape')
      nav.classList.remove('toc-open');
  });

  // Scoped to the pages that have a contents list, rather than imposed on
  // every page in the application
  document.documentElement.classList.add('toc-enabled');

  // Marks whichever section is currently on screen. The bottom margin keeps
  // the highlight on the section being read rather than the one below it.
  new bootstrap.ScrollSpy(document.body, {
    target: '#toc',
    rootMargin: '0px 0px -60%',
  });
});
