# Calendar, time and a small battery gauge, for the garage clock (AWTRIX NG on a TC002).
# Shown instead of the built-in Time app; install with clock/install.sh.
#
# Scripts on this build only get the small fonts, so the digits are the
# clock's own 3x5 glyphs (read back off its screen) drawn at double size,
# matching the built-in clock and notifications. The calendar page copies the
# built-in clock's icon pixel for pixel.

var DIGITS = [
  [7, 5, 5, 5, 7],  # 0   each row is 3 bits, left pixel = 4
  [2, 6, 2, 2, 7],  # 1
  [7, 1, 7, 4, 7],  # 2
  [7, 1, 7, 1, 7],  # 3
  [5, 5, 7, 1, 1],  # 4
  [7, 4, 7, 1, 7],  # 5
  [7, 4, 7, 5, 7],  # 6
  [7, 1, 1, 1, 1],  # 7
  [7, 5, 7, 5, 7],  # 8
  [7, 5, 7, 1, 7],  # 9
]

var TOP = 2          # time digits fill rows 2..11, like the built-in clock
var TIME_X = 17      # time sits right of the 16x16 calendar icon

class TimeBattery
  var volts       # one battery-voltage sample a minute, last 10 minutes
  var since       # loop() calls since the last sample
  var charging

  def init()
    self.volts = []
    self.since = 60
    self.charging = false
  end

  # The clock reports no charging flag, so infer it: a charger lifts the
  # battery voltage within a minute or two. Compare the newest minute with
  # three minutes earlier; rising = charging, falling = not.
  def loop()
    self.since += 1
    if self.since < 60 return end
    self.since = 0
    var v = sensor.battery_volts()
    if v == nil return end
    self.volts.push(v)
    if size(self.volts) > 10 self.volts.remove(0) end
    var n = size(self.volts)
    if n < 4 return end
    var delta = self.volts[n - 1] - self.volts[n - 4]
    if delta >= 0.02
      self.charging = true
    elif delta <= -0.005
      self.charging = false
    end
  end

  def glyph(x, y, rows, color)
    for r: 0 .. 4
      var bits = rows[r]
      for c: 0 .. 2
        if bits & (4 >> c) rect_fill(x + 2 * c, y + 2 * r, 2, 2, color) end
      end
    end
  end

  # The built-in clock's calendar page: red header, white page, day in black.
  def calendar(d)
    rect_fill(0, 0, 16, 2, 0xFF0000)
    rect_fill(0, 2, 16, 14, 0xFFFFFF)
    if d < 1 return end
    if d < 10
      self.glyph(5, 4, DIGITS[d], 0x000000)
    else
      self.glyph(1, 4, DIGITS[d / 10], 0x000000)
      self.glyph(9, 4, DIGITS[d % 10], 0x000000)
    end
  end

  def draw()
    clear()
    var white = 0xFFFFFF
    self.calendar(day())
    var h = hour()
    if h >= 0
      if !settings.get("time24h") h = (h + 11) % 12 + 1 end
      var m = minute()
      var digits = []
      if h >= 10 digits.push(h / 10) end
      digits.push(h % 10)
      var n_hour = size(digits)
      digits.push(m / 10)
      digits.push(m % 10)
      # Each digit is 6 px + 2 px gap; the colon is 2 px + 2 px gap.
      var w = size(digits) * 8 + 4 - 2
      var x = TIME_X + (width() - TIME_X - w) / 2
      for i: 0 .. size(digits) - 1
        if i == n_hour
          if second() % 2 == 0
            rect_fill(x, TOP + 2, 2, 2, white)
            rect_fill(x, TOP + 6, 2, 2, white)
          end
          x += 4
        end
        self.glyph(x, TOP, DIGITS[digits[i]], white)
        x += 8
      end
    end

    # Battery: a small cell under the time, right-aligned, filled left to right.
    var b = sensor.battery()
    if b != nil
      var color = b > 50 ? 0x00C000 : (b > 20 ? 0xFFD000 : 0xFF0000)
      var grey = 0x606060
      rect(40, 13, 11, 3, grey)
      pixel(51, 14, grey)
      var fill = (9 * b + 50) / 100
      if fill > 0 line(41, 14, 40 + fill, 14, color) end
      if self.charging rect_fill(37, 13, 2, 2, 0x00FF00) end   # charging dot
    end
  end
end

return TimeBattery()
