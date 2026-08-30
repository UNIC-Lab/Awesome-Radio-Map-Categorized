#!/usr/bin/env ruby

ROOT = File.expand_path("..", __dir__)
README_PATH = File.join(ROOT, "README.md")

lines = File.readlines(README_PATH, chomp: true)
errors = []
entries = []
summaries = []
mode = nil
section = nil
group = nil

lines.each_with_index do |line, index|
  mode = "with_code" if line == "## Papers with Code"
  mode = "without_code" if line == "## Papers without Code"
  section = Regexp.last_match(1).downcase if line =~ /^### (Journals|Conferences|Preprints)$/

  if line =~ /^<summary><strong>(.+?)<\/strong> <sub>(\d+) (papers?)<\/sub><\/summary>$/
    group = Regexp.last_match(1)
    summaries << {
      line: index + 1,
      mode: mode,
      section: section,
      group: group,
      declared: Regexp.last_match(2).to_i
    }
  end

  next unless line =~ /^- \*\*(.+?)\*\*(?:<br>|  )$/

  title = Regexp.last_match(1)
  metadata = lines[index + 1].to_s
  unless metadata =~ /^  \*(.+) ((?:19|20)\d{2})\*/
    errors << "README.md:#{index + 2}: malformed or missing venue metadata for #{title}"
    next
  end

  entries << {
    line: index + 1,
    title: title,
    metadata: metadata,
    mode: mode,
    section: section,
    group: group,
    venue: Regexp.last_match(1),
    year: Regexp.last_match(2).to_i,
    paper_url: metadata[/\[Paper\]\(([^)]+)\)/, 1],
    ranking: metadata[/<sub>([^<]+)<\/sub>/, 1],
    has_code: metadata.include?("[Code]("),
    has_repository: metadata.include?("[Repository]("),
    has_dataset: metadata.include?("[Dataset](")
  }
end

errors << "README.md: expected 383 entries, found #{entries.length}" unless entries.length == 383

entries.each do |entry|
  errors << "README.md:#{entry[:line]}: missing paper URL" unless entry[:paper_url]

  if entry[:mode] == "with_code" && !entry[:has_code]
    errors << "README.md:#{entry[:line]}: Papers with Code entry has no Code link"
  elsif entry[:mode] == "without_code" && entry[:has_code]
    errors << "README.md:#{entry[:line]}: Papers without Code entry contains a Code link"
  end

  if entry[:has_repository] || entry[:has_dataset]
    statuses = ["Code forthcoming", "Dataset only", "Repository placeholder", "Empty repository"]
    unless statuses.any? { |status| entry[:metadata].include?("`#{status}`") }
      errors << "README.md:#{entry[:line] + 1}: repository or dataset link is missing a recognized status"
    end
  end

  if entry[:section] == "journals"
    errors << "README.md:#{entry[:line]}: journal is missing ranking metadata" unless entry[:ranking]
    if entry[:ranking] && entry[:ranking] !~ /^JCR [^·]+ · 中科院分区 [^·]+ · IF [^·]+$/
      errors << "README.md:#{entry[:line]}: malformed journal ranking metadata"
    end
  elsif entry[:ranking]
    errors << "README.md:#{entry[:line]}: conference or preprint has journal ranking metadata"
  end

  if entry[:group] =~ /^\d{4}$/ && entry[:year] != entry[:group].to_i
    errors << "README.md:#{entry[:line]}: year #{entry[:year]} does not match group #{entry[:group]}"
  elsif entry[:group] == "2024 and Earlier" && entry[:year] > 2024
    errors << "README.md:#{entry[:line]}: year #{entry[:year]} is in the 2024-and-earlier group"
  end
end

[:title, :paper_url].each do |field|
  entries.group_by { |entry| entry[field] }.each_value do |matches|
    next if matches.first[field].nil? || matches.length == 1

    errors << "README.md: duplicate #{field} at lines #{matches.map { |entry| entry[:line] }.join(', ')}"
  end
end

def title_sort_key(title)
  title.downcase.gsub(/\$|\\|\^|\{|\}/, " ").gsub(/[^a-z0-9]+/, " ").strip
end

entries.group_by { |entry| [entry[:mode], entry[:section], entry[:group]] }.each_value do |block|
  block.each_cons(2) do |left, right|
    next if title_sort_key(left[:title]) <= title_sort_key(right[:title])

    errors << "README.md:#{right[:line]}: title is not alphabetized after line #{left[:line]}"
  end
end

summaries.each do |summary|
  actual = entries.count do |entry|
    [entry[:mode], entry[:section], entry[:group]] == [summary[:mode], summary[:section], summary[:group]]
  end
  errors << "README.md:#{summary[:line]}: declares #{summary[:declared]} papers, found #{actual}" unless actual == summary[:declared]
end

counts = Hash.new(0)
entries.each { |entry| counts[[entry[:mode], entry[:section]]] += 1 }
browse_rows = {}
lines.each do |line|
  next unless line =~ /^\| \[(With code|Without code)\].*?\| (\d+) \| (\d+) \| (\d+) \| \*\*(\d+)\*\* \|$/

  browse_rows[Regexp.last_match(1)] = [2, 3, 4, 5].map { |number| Regexp.last_match(number).to_i }
end

{
  "With code" => "with_code",
  "Without code" => "without_code"
}.each do |label, key|
  actual = %w[journals conferences preprints].map { |name| counts[[key, name]] }
  actual << actual.inject(0, :+)
  errors << "README.md: Browse row #{label} is #{browse_rows[label].inspect}, expected #{actual.inspect}" unless browse_rows[label] == actual
end

all_actual = %w[journals conferences preprints].map do |name|
  counts[["with_code", name]] + counts[["without_code", name]]
end
all_actual << all_actual.inject(0, :+)
all_line = lines.find { |line| line.start_with?("| **All papers**") }
all_declared = all_line&.scan(/\*\*(\d+)\*\*/)&.flatten&.map(&:to_i)
errors << "README.md: All papers row is #{all_declared.inspect}, expected #{all_actual.inspect}" unless all_declared == all_actual

venue_rankings = {}
entries.select { |entry| entry[:section] == "journals" }.each do |entry|
  if venue_rankings.key?(entry[:venue]) && venue_rankings[entry[:venue]] != entry[:ranking]
    errors << "README.md:#{entry[:line]}: inconsistent ranking for #{entry[:venue]}"
  else
    venue_rankings[entry[:venue]] = entry[:ranking]
  end
end

if errors.empty?
  puts "Catalog check passed: #{entries.length} papers, #{entries.count { |entry| entry[:has_code] }} code releases, #{venue_rankings.length} journal venues."
  exit 0
end

warn errors.join("\n")
exit 1
