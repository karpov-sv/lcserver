/* Table of contents for the target page.
 *
 * Built from the rendered page rather than from the survey registry: which
 * sections a target actually shows depends on its coordinates, on what has
 * been acquired already and on whether a chain is running, so a list built
 * from anything else would drift and end up pointing at sections that are
 * not there.
 */

// How a source's state is drawn beside its entry. Font Awesome carries the
// spin for us, so a running step needs no animation of its own here.
var TOC_STATES = {
  pending: {icon: 'fa-clock-o',           css: 'text-secondary', title: 'waiting for a worker'},
  running: {icon: 'fa-spinner fa-spin',   css: 'text-primary',   title: 'running'},
  done:    {icon: 'fa-check',             css: 'text-success',   title: 'done'},
  // Ran, and there was nothing to be had - which is not a failure, and is
  // drawn quietly so that the sources carrying data are what stands out
  empty:   {icon: 'fa-minus',             css: 'text-secondary opacity-50', title: 'no data'},
  failed:  {icon: 'fa-times',             css: 'text-danger',    title: 'failed'},
};


function tocApplyState(mark, state) {
  var shown = TOC_STATES[state];

  mark.className = 'toc-state fa ' + (shown ? shown.icon + ' ' + shown.css : '');
  mark.title = shown ? shown.title : '';
}


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

  // The kind the last entry belonged to, so that the rail can be ruled where
  // it changes. Only between two kinds, not before the first: the sections
  // above the sources - the info step - carry no kind at all, and a line
  // under them would be marking the wrong thing.
  var lastGroup = null;

  headings.forEach(function(heading) {
    var group = heading.dataset.tocGroup;

    if (group && lastGroup && group !== lastGroup) {
      var rule = document.createElement('li');
      rule.className = 'toc-rule';
      rule.setAttribute('aria-hidden', 'true');
      list.appendChild(rule);
    }

    if (group)
      lastGroup = group;

    var item = document.createElement('li');
    item.className = 'nav-item';

    var link = document.createElement('a');
    link.className = 'nav-link';
    link.href = '#' + heading.id;
    link.title = heading.dataset.tocTitle || heading.textContent.trim();

    // The short name, with the full one left for the tooltip - the rail is
    // deliberately too narrow for 'Mini-MegaTORTORA'
    var label = document.createElement('span');
    label.className = 'toc-label';
    label.textContent = heading.dataset.toc || heading.textContent.trim();
    link.appendChild(label);

    // How the source stands, which the page keeps up to date as it polls.
    // It lives here rather than beside the heading so that it is in view
    // whichever part of the page is being read.
    var mark = document.createElement('i');
    if (heading.dataset.tocSource)
      mark.dataset.source = heading.dataset.tocSource;
    tocApplyState(mark, heading.dataset.tocState);
    link.appendChild(mark);

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

  nav.addEventListener('click', function(event) {
    var link = event.target.closest('.nav-link');

    if (!link)
      return;

    // Opened over the page, it should get out of the way once it has been used
    nav.classList.remove('toc-open');

    // Let the browser have the ones meant for a new tab or window
    if (event.button !== 0 || event.metaKey || event.ctrlKey
        || event.shiftKey || event.altKey)
      return;

    var section = document.getElementById(link.getAttribute('href').slice(1));

    if (!section)
      return;

    // Scrolled to rather than navigated to, so the address keeps no record of
    // it: following the link as a link would leave the section in the URL, and
    // every later reload of the page would jump back down to it. The offset
    // that clears the sticky header is scroll-margin-top, in the stylesheet.
    event.preventDefault();
    section.scrollIntoView({behavior: 'smooth', block: 'start'});

    // And it is the one marked, whatever the scrolling ends up showing
    chosen = link;
    scheduleMark();
  });

  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape')
      nav.classList.remove('toc-open');
  });

  // The page hands over each source's state as it polls for it
  window.tocSetStates = function(states) {
    nav.querySelectorAll('.toc-state[data-source]').forEach(function(mark) {
      tocApplyState(mark, states[mark.dataset.source]);
    });
  };

  // Marks the section being read: the last one whose heading has passed the
  // upper part of the window. Not Bootstrap's ScrollSpy, which only marks a
  // section while its heading itself is in view, and so loses it as soon as
  // the reading goes past the form into a long log or the plots below.
  var links = nav.querySelectorAll('.nav-link');
  var pending = false;

  // The entry just followed, kept marked until the reader scrolls by hand. A
  // section near the end of the page may not be scrollable up to the line, or
  // even to the top - the page ends first - and would otherwise show as the
  // one next to it.
  var chosen = null;

  ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(function(type) {
    window.addEventListener(type, function(event) {
      // Not a click on the rail itself, which is what set it
      if (chosen && !(event.target instanceof Node && nav.contains(event.target))) {
        chosen = null;
        scheduleMark();
      }
    }, {passive: true});
  });

  function markCurrent() {
    pending = false;

    // The last sections may never get their heading up to the line, as the
    // page ends before they could. So over the last stretch of scrolling the
    // line slides down to the bottom of the window, passing their headings in
    // turn and reaching the very last one as the page ends.
    var height = window.innerHeight;
    var left = document.documentElement.scrollHeight - window.scrollY - height;
    var line = height * 0.4 + Math.max(0, height * 0.6 - left);
    var current = -1;

    headings.forEach(function(heading, i) {
      if (heading.getBoundingClientRect().top <= line)
        current = i;
    });

    links.forEach(function(link, i) {
      link.classList.toggle('active', chosen ? link === chosen : i === current);
    });
  }

  function scheduleMark() {
    if (!pending) {
      pending = true;
      window.requestAnimationFrame(markCurrent);
    }
  }

  window.addEventListener('scroll', scheduleMark, {passive: true});
  window.addEventListener('resize', scheduleMark);
  // A log opened or closed, or a plot loaded, moves the sections without any
  // scrolling
  new ResizeObserver(scheduleMark).observe(document.body);
  markCurrent();
});
