# Time with a small battery gauge, for the garage clock (AWTRIX NG on a TC002).
# Shown instead of the built-in Time app; install with clock/install.sh.
#
# Scripts on this build only get the small fonts, so the digits are the
# clock's own 3x5 glyphs (read back off its screen) drawn at double size,
# matching the built-in clock and notifications.

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

var TOP = 2          # digits fill rows 2..11, like the built-in clock
var TIME_AREA = 46   # columns 0..45; the battery gauge sits at 48..51

class TimeBattery
  def glyph(x, rows, color)
    for r: 0 .. 4
      var bits = rows[r]
      for c: 0 .. 2
        if bits & (4 >> c) rect_fill(x + 2 * c, TOP + 2 * r, 2, 2, color) end
      end
    end
  end

  def draw()
    clear()
    var h = hour()
    var white = 0xFFFFFF
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
      var x = (TIME_AREA - w) / 2
      for i: 0 .. size(digits) - 1
        if i == n_hour
          if second() % 2 == 0
            rect_fill(x, TOP + 2, 2, 2, white)
            rect_fill(x, TOP + 6, 2, 2, white)
          end
          x += 4
        end
        self.glyph(x, DIGITS[digits[i]], white)
        x += 8
      end
    end

    # Battery: a cell at the right edge, filled bottom-up by charge.
    var b = sensor.battery()
    if b != nil
      var color = b > 50 ? 0x00C000 : (b > 20 ? 0xFFD000 : 0xFF0000)
      var grey = 0x606060
      line(49, TOP, 50, TOP, grey)
      rect(48, TOP + 1, 4, 9, grey)
      var fill = (7 * b + 50) / 100
      if fill > 0 rect_fill(49, TOP + 9 - fill, 2, fill, color) end
    end
  end
end

return TimeBattery()
