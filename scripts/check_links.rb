#!/usr/bin/env ruby

require "net/http"
require "openssl"
require "thread"
require "uri"

readme = File.read(File.expand_path("../README.md", __dir__))
urls = readme.scan(/\[Paper\]\((https?:\/\/[^)]+)\)/).flatten.uniq
queue = Queue.new
urls.each { |url| queue << url }
results = []
mutex = Mutex.new

def fetch_status(url, method, redirects = 8)
  raise "too many redirects" if redirects.zero?

  uri = URI(url)
  http_request = method == :head ? Net::HTTP::Head.new(uri) : Net::HTTP::Get.new(uri)
  http_request["User-Agent"] = "Awesome-Radio-Map-Link-Check/1.0"
  response = Net::HTTP.start(
    uri.host,
    uri.port,
    use_ssl: uri.scheme == "https",
    open_timeout: 12,
    read_timeout: 20
  ) { |http| http.request(http_request) }

  if response.is_a?(Net::HTTPRedirection) && response["location"]
    return fetch_status(URI.join(url, response["location"]).to_s, method, redirects - 1)
  end
  response.code.to_i
end

workers = 12.times.map do
  Thread.new do
    loop do
      url = queue.pop(true)
      status = fetch_status(url, :head)
      status = fetch_status(url, :get) if [400, 403, 405, 429].include?(status) || status >= 500
      mutex.synchronize { results << [url, status, nil] }
    rescue ThreadError
      break
    rescue StandardError => e
      mutex.synchronize { results << [url, nil, "#{e.class}: #{e.message}"] }
    end
  end
end
workers.each(&:join)

dead = results.select { |_url, status, _error| [404, 410].include?(status) }
warnings = results.select do |_url, status, error|
  error || (!status.nil? && !(200..399).cover?(status) && ![401, 403, 429].include?(status) && ![404, 410].include?(status))
end
blocked = results.count { |_url, status, _error| [401, 403, 429].include?(status) }

dead.each { |url, status, _error| warn "DEAD #{status} #{url}" }
warnings.each { |url, status, error| warn "WARN #{status || error} #{url}" }
puts "Checked #{urls.length} paper links: #{dead.length} dead, #{blocked} access-blocked, #{warnings.length} warnings."
exit(dead.empty? ? 0 : 1)
