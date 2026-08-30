#!/usr/bin/env ruby

require "json"
require "net/http"
require "uri"

token = ENV["GITHUB_TOKEN"].to_s
abort "GITHUB_TOKEN is required for the GitHub repository audit." if token.empty?

readme = File.read(File.expand_path("../README.md", __dir__))
repositories = readme.scan(/\[Code\]\(https:\/\/github\.com\/([^)]+)\)/).flatten.map do |path|
  path.sub(/\.git\z/, "").split("/").first(2).join("/")
end.uniq

SOURCE_EXTENSIONS = %w[.py .ipynb .m .mlx .cpp .cc .c .h .hpp .cu .cuh .java .jl .r .rs .go .js .ts .tsx .sh].freeze
IGNORED_PREFIXES = %w[docs/ assets/ images/ figures/ .github/].freeze

def github_json(path, token)
  uri = URI("https://api.github.com/#{path}")
  request = Net::HTTP::Get.new(uri)
  request["Authorization"] = "Bearer #{token}"
  request["Accept"] = "application/vnd.github+json"
  request["X-GitHub-Api-Version"] = "2022-11-28"
  request["User-Agent"] = "Awesome-Radio-Map-Code-Check/1.0"
  response = Net::HTTP.start(uri.host, uri.port, use_ssl: true, open_timeout: 12, read_timeout: 30) do |http|
    http.request(request)
  end
  raise "GitHub API #{response.code} for #{path}" unless response.is_a?(Net::HTTPSuccess)

  JSON.parse(response.body)
end

failures = []
repositories.each do |repository|
  metadata = github_json("repos/#{repository}", token)
  branch = metadata.fetch("default_branch")
  tree = github_json("repos/#{repository}/git/trees/#{URI.encode_www_form_component(branch)}?recursive=1", token)
  source_files = tree.fetch("tree", []).each_with_object([]) do |item, found|
    next unless item["type"] == "blob"

    path = item.fetch("path")
    next if IGNORED_PREFIXES.any? { |prefix| path.downcase.start_with?(prefix) }
    next unless SOURCE_EXTENSIONS.include?(File.extname(path).downcase)

    found << path
  end
  failures << repository if source_files.empty?
  puts "#{repository}: #{source_files.length} source/notebook files"
end

unless failures.empty?
  warn "Repositories linked as Code but containing no source/notebook: #{failures.join(', ')}"
  exit 1
end

puts "Code repository check passed: #{repositories.length} repositories contain an implementation."
