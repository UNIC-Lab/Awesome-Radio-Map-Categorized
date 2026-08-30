#!/usr/bin/env ruby

readme = File.read(File.expand_path("../README.md", __dir__))

forbidden = {
  "duplicated JCR label" => /<sub>[^<]*JCR[^<]*JCR[^<]*<\/sub>/,
  "known title typo: Deep Leaning" => /Deep Leaning/,
  "known title corruption" => /BasElectriacaled/,
  "known venue typo" => /Tutorialsls/,
  "legacy IoT Journal abbreviation" => /IEEE IoT J/,
  "legacy TWC abbreviation" => /IEEE TWC/,
  "legacy TVT abbreviation" => /IEEE TVT/,
  "legacy TCCN abbreviation" => /IEEE TCCN/,
  "legacy GLOBECOM spelling" => /IEEE GlobeCom/,
  "generic ACM venue" => /\*ACM 20\d{2}\*/,
  "legacy workshop abbreviation" => /\b(?:wksp|ICASSPW)\b/i,
  "legacy VTC formatting" => /VTC20\d{2}-(?:Spring|Fall)/
}

errors = forbidden.each_with_object([]) do |(label, pattern), found|
  match = readme.match(pattern)
  next unless match

  line = readme[0...match.begin(0)].count("\n") + 1
  found << "README.md:#{line}: #{label}: #{match[0]}"
end

section = nil
readme.each_line.with_index(1) do |line, number|
  section = Regexp.last_match(1) if line =~ /^### (Journals|Conferences|Preprints)$/
  next unless line =~ /^  \*([^*]+)\* ·/
  venue = Regexp.last_match(1)
  errors << "README.md:#{number}: venue is missing a publication year: #{venue}" unless venue =~ /(?:19|20)\d{2}/

  has_ranking = line =~ /<sub>[^<]*(?:JCR|中科院|IF|CCF)[^<]*<\/sub>/
  if section == "Preprints" && has_ranking
    errors << "README.md:#{number}: preprint should not have venue ranking metadata: #{venue}"
  elsif section != "Preprints" && !has_ranking
    errors << "README.md:#{number}: journal or conference is missing ranking metadata: #{venue}"
  end
end

if errors.empty?
  puts "README metadata lint passed."
  exit 0
end

warn errors.join("\n")
exit 1
