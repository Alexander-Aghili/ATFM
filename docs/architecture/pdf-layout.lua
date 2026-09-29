local section = ''
local root = 'https://github.com/Alexander-Aghili/ATFM/blob/3084f34/'

function Link(link)
  if link.target == 'atfm-architecture.pdf' then
    link.target = root:gsub('3084f34', 'main') .. 'docs/architecture/atfm-architecture.pdf'
    return link
  end
  if not link.target:match('^https?://') and not link.target:match('^#') then
    local parts = {'docs', 'architecture'}
    for part in link.target:gmatch('[^/]+') do
      if part == '..' then table.remove(parts)
      elseif part ~= '.' then table.insert(parts, part) end
    end
    local target = table.concat(parts, '/')
    local base = target:match('^docs/') and root:gsub('3084f34', 'main') or root
    link.target = base .. target
  end
  return link
end

function Header(header)
  if header.level == 1 then return {} end
  if header.level == 3 and pandoc.utils.stringify(header) == 'Planning and delivery' then
    return {pandoc.RawBlock('latex', '\\Needspace{12\\baselineskip}'), header}
  end
  if header.level == 2 and pandoc.utils.stringify(header):match('^%d') then
    section = pandoc.write(pandoc.Pandoc({header}), 'latex')
    return {}
  end
end

function Figure(figure)
  local image = figure.content[1].content[1]
  local file = image.src:match('([^/]+)%.svg$')
  if not file then return figure end
  return pandoc.RawBlock('latex', '\\clearpage\\begin{landscape}\n'
    .. section .. '\\thispagestyle{plain}\\centering\n'
    .. '\\vspace*{0.3cm}\\includegraphics[width=\\linewidth,height=0.82\\textheight,keepaspectratio]{'
    .. 'tmp/pdfs/architecture/' .. file .. '.pdf}\n'
    .. '\\par\\medskip{\\small ' .. pandoc.utils.stringify(figure.caption):gsub('[&%%#_]', '\\%0')
    .. '}\\end{landscape}\\clearpage')
end

function Table(tbl)
  if #tbl.colspecs == 3 then
    local widths = {0.28, 0.27, 0.45}
    local endpoints = pandoc.utils.stringify(tbl.head.rows[1].cells[2].contents):match('Endpoint')
    if endpoints then widths = {0.10, 0.48, 0.42} end
    for i, spec in ipairs(tbl.colspecs) do spec[2] = widths[i] end
    if endpoints then
      return {pandoc.RawBlock('latex', '\\begingroup\\small'), tbl, pandoc.RawBlock('latex', '\\endgroup')}
    end
  end
  return tbl
end
